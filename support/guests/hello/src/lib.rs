//! Greeting component used to verify the guest build and host harness.
//! It has no imports and returns a deterministic greeting for its caller.

#![allow(clippy::same_length_and_capacity)]

wit_bindgen::generate!({
    path: "wit",
    world: "hello",
});

struct Component;

impl Guest for Component {
    fn greet(name: String) -> String {
        format!("Hello, {name}!")
    }
}

export!(Component);
