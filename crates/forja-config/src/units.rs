use std::{fmt, marker::PhantomData, num::IntErrorKind, time};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de, de::IntoDeserializer};

/// A byte count serialized as an integer or an IEC quantity such as `8GiB`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteSize(u64);

impl ByteSize {
    /// Creates a byte count.
    #[must_use]
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// Returns the byte count.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Serialize for ByteSize {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ByteSizeVisitor)
    }
}

struct ByteSizeVisitor;

impl de::Visitor<'_> for ByteSizeVisitor {
    type Value = ByteSize;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a non-negative byte count or IEC byte quantity")
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(ByteSize(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        u64::try_from(value)
            .map(ByteSize)
            .map_err(|_| E::custom(format!("quantity {value} overflows u64")))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        parse_quantity(value, BYTE_UNITS)
            .map(ByteSize)
            .map_err(E::custom)
    }
}

const BYTE_UNITS: &[(&str, u64)] = &[
    ("TiB", 1024_u64.pow(4)),
    ("GiB", 1024_u64.pow(3)),
    ("MiB", 1024_u64.pow(2)),
    ("KiB", 1024),
    ("B", 1),
];

/// A duration serialized with an integer value and unit suffix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Duration(time::Duration);

impl Duration {
    /// Creates a duration from milliseconds.
    #[must_use]
    pub const fn from_millis(milliseconds: u64) -> Self {
        Self(time::Duration::from_millis(milliseconds))
    }

    /// Creates a duration from seconds.
    #[must_use]
    pub const fn from_secs(seconds: u64) -> Self {
        Self(time::Duration::from_secs(seconds))
    }

    /// Returns the standard-library duration.
    #[must_use]
    pub const fn get(self) -> time::Duration {
        self.0
    }
}

impl Serialize for Duration {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{}ms", self.0.as_millis()))
    }
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        parse_quantity(&value, DURATION_UNITS)
            .map(time::Duration::from_millis)
            .map(Self)
            .map_err(de::Error::custom)
    }
}

const DURATION_UNITS: &[(&str, u64)] = &[("ms", 1), ("s", 1_000), ("m", 60_000), ("h", 3_600_000)];

fn parse_quantity(value: &str, units: &[(&str, u64)]) -> Result<u64, String> {
    let Some((digits, multiplier)) = units
        .iter()
        .find_map(|&(suffix, multiplier)| value.strip_suffix(suffix).map(|v| (v, multiplier)))
    else {
        return match value.parse::<u64>() {
            Err(error) if matches!(error.kind(), IntErrorKind::PosOverflow) => {
                Err(format!("quantity {value:?} overflows u64"))
            }
            _ => Err(format!("invalid quantity {value:?}")),
        };
    };
    let count = match digits.parse::<u64>() {
        Ok(count) => count,
        Err(error) if matches!(error.kind(), IntErrorKind::PosOverflow) => {
            return Err(format!("quantity {value:?} overflows u64"));
        }
        Err(_) => return Err(format!("invalid quantity {value:?}")),
    };
    count
        .checked_mul(multiplier)
        .ok_or_else(|| format!("quantity {value:?} overflows u64"))
}

/// A finite value or an explicit absence of a limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unbounded<T> {
    /// No limit is imposed.
    Unlimited,
    /// A finite limit.
    Limited(T),
}

impl<T> Serialize for Unbounded<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Unlimited => serializer.serialize_str("unlimited"),
            Self::Limited(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T> Deserialize<'de> for Unbounded<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UnboundedVisitor(PhantomData))
    }
}

struct UnboundedVisitor<T>(PhantomData<T>);

impl<'de, T> de::Visitor<'de> for UnboundedVisitor<T>
where
    T: Deserialize<'de>,
{
    type Value = Unbounded<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("\"unlimited\" or a finite value")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value == "unlimited" {
            Ok(Unbounded::Unlimited)
        } else {
            T::deserialize(value.into_deserializer()).map(Unbounded::Limited)
        }
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        T::deserialize(value.into_deserializer()).map(Unbounded::Limited)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        T::deserialize(value.into_deserializer()).map(Unbounded::Limited)
    }
}

#[cfg(test)]
mod tests {
    use serde::de::value::{Error, I64Deserializer, StrDeserializer, U64Deserializer};

    use super::*;

    #[test]
    fn parses_byte_sizes_with_checked_arithmetic() {
        let value = ByteSize::deserialize(StrDeserializer::<Error>::new("8GiB")).unwrap();
        assert_eq!(value.get(), 8 * 1024 * 1024 * 1024);
        let integer = ByteSize::deserialize(U64Deserializer::<Error>::new(4097)).unwrap();
        assert_eq!(integer.get(), 4097);
        let zero = ByteSize::deserialize(U64Deserializer::<Error>::new(0)).unwrap();
        assert_eq!(zero.get(), 0);
        assert_eq!(
            ByteSize::deserialize(StrDeserializer::<Error>::new("18446744073709551615GiB"))
                .unwrap_err()
                .to_string(),
            "quantity \"18446744073709551615GiB\" overflows u64"
        );
        assert_eq!(
            ByteSize::deserialize(I64Deserializer::<Error>::new(-1))
                .unwrap_err()
                .to_string(),
            "quantity -1 overflows u64"
        );
        for value in ["1.5GiB", " 1GiB", "1GiB "] {
            assert_eq!(
                ByteSize::deserialize(StrDeserializer::<Error>::new(value))
                    .unwrap_err()
                    .to_string(),
                format!("invalid quantity {value:?}")
            );
        }
        let value = "18446744073709551616";
        assert_eq!(
            ByteSize::deserialize(StrDeserializer::<Error>::new(value))
                .unwrap_err()
                .to_string(),
            format!("quantity {value:?} overflows u64")
        );
    }

    #[test]
    fn parses_duration_units_with_checked_arithmetic() {
        for (value, milliseconds) in [
            ("0s", 0),
            ("250ms", 250),
            ("300s", 300_000),
            ("60m", 3_600_000),
        ] {
            let duration = Duration::deserialize(StrDeserializer::<Error>::new(value)).unwrap();
            assert_eq!(duration.get(), time::Duration::from_millis(milliseconds));
        }
        let value = "18446744073709551615h";
        assert_eq!(
            Duration::deserialize(StrDeserializer::<Error>::new(value))
                .unwrap_err()
                .to_string(),
            format!("quantity {value:?} overflows u64")
        );
    }

    #[test]
    fn parses_unbounded_values() {
        assert_eq!(
            Unbounded::<u32>::deserialize(StrDeserializer::<Error>::new("unlimited")).unwrap(),
            Unbounded::Unlimited
        );
        assert_eq!(
            Unbounded::<u32>::deserialize(U64Deserializer::<Error>::new(4097)).unwrap(),
            Unbounded::Limited(4097)
        );
    }
}
