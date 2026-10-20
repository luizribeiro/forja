{ lib }:
let
  inherit (lib) mkOption types;

  checkedInteger =
    schema:
    types.addCheck types.ints.unsigned (
      value:
      (!schema ? minimum || value >= schema.minimum) && (!schema ? maximum || value <= schema.maximum)
    );

  stringType =
    schema:
    if schema ? "x-forja-absolute-path" then
      types.strMatching "^/.*"
    else if schema ? pattern then
      types.strMatching schema.pattern
    else
      types.str;

  typeFor =
    schema:
    if schema ? oneOf then
      types.oneOf (map typeFor schema.oneOf)
    else if schema ? const then
      types.enum [ schema.const ]
    else if builtins.isList (schema.type or null) then
      let
        concrete = builtins.filter (kind: kind != "null") schema.type;
      in
      if builtins.length concrete == 1 && builtins.elem "null" schema.type then
        types.nullOr (typeFor (schema // { type = builtins.head concrete; }))
      else
        throw "unsupported JSON Schema type union"
    else if (schema.type or null) == "integer" then
      checkedInteger schema
    else if (schema.type or null) == "string" then
      stringType schema
    else if (schema.type or null) == "boolean" then
      types.bool
    else if (schema.type or null) == "object" then
      types.submodule { options = optionsFor schema; }
    else
      throw "unsupported JSON Schema node";

  optionFor =
    schema:
    mkOption (
      {
        type = typeFor schema;
        description = schema.description or "";
      }
      // lib.optionalAttrs (schema ? default) { inherit (schema) default; }
    );

  optionsFor = schema: lib.mapAttrs (_: optionFor) schema.properties;
in
optionsFor
