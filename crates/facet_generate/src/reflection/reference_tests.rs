//! Every reference to a type names the type as it is registered, whatever wraps the reference
//! (#160, #167), and `build` rejects a reference that names no registered type.

use std::collections::BTreeMap;

use facet::Facet;

use crate as fg;
use crate::{
    Registry,
    error::Error,
    reflect,
    reflection::{
        RegistryBuilder,
        format::{
            ContainerFormat, Doc, Format, FormatHolder, Named, QualifiedTypeName, VariantFormat,
        },
    },
};

/// Every reference in `registry` to a type called `name`, as `(referring container, location,
/// the name referred to)`.
fn references_to(registry: &Registry, name: &str) -> Vec<(String, String, QualifiedTypeName)> {
    let mut found = vec![];
    let mut collect = |container: &QualifiedTypeName, location: String, format: &Format| {
        format
            .visit(&mut |format| {
                if let Format::TypeName(qualified) = format
                    && qualified.name == name
                {
                    found.push((container.to_string(), location.clone(), qualified.clone()));
                }
                Ok(())
            })
            .unwrap();
    };
    for (container_name, container) in registry {
        match container {
            ContainerFormat::UnitStruct(_) => {}
            ContainerFormat::NewTypeStruct(format, _) => {
                collect(container_name, "0".to_string(), format);
            }
            ContainerFormat::TupleStruct(formats, _) => {
                for (index, format) in formats.iter().enumerate() {
                    collect(container_name, index.to_string(), format);
                }
            }
            ContainerFormat::Struct(fields, _) => {
                for field in fields {
                    collect(container_name, field.name.clone(), &field.value);
                }
            }
            ContainerFormat::Enum(variants, _, _) => {
                for variant in variants.values() {
                    match &variant.value {
                        VariantFormat::Variable(_) | VariantFormat::Unit => {}
                        VariantFormat::NewType(format) => {
                            collect(container_name, variant.name.clone(), format);
                        }
                        VariantFormat::Tuple(formats) => {
                            for (index, format) in formats.iter().enumerate() {
                                collect(
                                    container_name,
                                    format!("{}.{index}", variant.name),
                                    format,
                                );
                            }
                        }
                        VariantFormat::Struct(fields) => {
                            for field in fields {
                                let location = format!("{}.{}", variant.name, field.name);
                                collect(container_name, location, &field.value);
                            }
                        }
                    }
                }
            }
        }
    }
    found
}

/// Asserts that `Leaf` is registered once, as `expected`, and that every reference to it, from
/// every position in [`holders!`], names it as `expected`. `leaf_in_root` is whether the ROOT
/// `App` holds a `Leaf` too.
fn assert_every_reference_names(
    registry: &Registry,
    expected: &QualifiedTypeName,
    leaf_in_root: bool,
) {
    let registered: Vec<_> = registry.keys().filter(|key| key.name == "Leaf").collect();
    assert_eq!(
        registered,
        [expected],
        "`Leaf` registered as {registered:?}"
    );

    let references = references_to(registry, "Leaf");
    let wrong: Vec<_> = references
        .iter()
        .filter(|(_, _, name)| name != expected)
        .collect();
    assert!(
        wrong.is_empty(),
        "references not naming {expected}: {wrong:#?}"
    );

    let mut locations: Vec<_> = references
        .iter()
        .map(|(container, location, _)| format!("{container}.{location}"))
        .collect();
    locations.sort();
    let mut expected_locations = vec![
        "detail::Holder.array",
        "detail::Holder.direct",
        "detail::Holder.map_key",
        "detail::Holder.map_value",
        "detail::Holder.option",
        "detail::Holder.option_seq",
        "detail::Holder.seq",
        "detail::Holder.seq_option",
        "detail::Holder.set",
        "detail::Holder.transparent",
        "detail::Holder.transparent_seq",
        "detail::Holder.tuple",
        "detail::Pair.0",
        "detail::Pair.1",
        "detail::Payload.NewType",
        "detail::Payload.NewTypeSeq",
        "detail::Payload.Struct.map",
        "detail::Payload.Struct.option",
        "detail::Payload.Tuple.0",
        "detail::Payload.Tuple.1",
        "detail::Wrapped.0",
    ];
    if leaf_in_root {
        expected_locations.insert(0, "ROOT::App.leaf_in_root");
    }
    assert_eq!(locations, expected_locations);
}

/// Types in namespace `detail` referring to `Leaf` from every kind of position, each wrapping it
/// differently, and a ROOT `App` holding them. `$leaf` is `Leaf`'s own namespace attribute, if
/// any, and `$leaf_in_root` names a field of `App` holding a `Leaf` directly. (An unannotated
/// `Leaf` held by `App` would be registered in ROOT as well as in `detail`.)
macro_rules! holders {
    ($(#[$leaf:meta])* ; $($leaf_in_root:ident)?) => {
        #[derive(Facet, PartialEq, Eq, PartialOrd, Ord)]
        $(#[$leaf])*
        pub struct Leaf {
            pub id: u32,
        }

        #[derive(Facet)]
        #[facet(transparent)]
        pub struct LeafId(Leaf);

        #[derive(Facet)]
        #[facet(fg::namespace = "detail")]
        pub struct Holder {
            pub direct: Leaf,
            pub option: Option<Leaf>,
            pub seq: Vec<Leaf>,
            pub option_seq: Option<Vec<Leaf>>,
            pub seq_option: Vec<Option<Leaf>>,
            pub set: std::collections::BTreeSet<Leaf>,
            pub map_key: BTreeMap<Leaf, u8>,
            pub map_value: BTreeMap<u8, Leaf>,
            pub tuple: (Leaf, u8),
            pub array: [Leaf; 2],
            pub transparent: LeafId,
            pub transparent_seq: Vec<LeafId>,
        }

        #[derive(Facet)]
        #[repr(C)]
        #[facet(fg::namespace = "detail")]
        #[allow(dead_code)]
        pub enum Payload {
            NewType(Option<Leaf>),
            NewTypeSeq(Vec<Leaf>),
            Tuple(Option<Leaf>, Vec<(u8, Leaf)>),
            Struct {
                option: Option<Leaf>,
                map: BTreeMap<u8, Vec<Leaf>>,
            },
        }

        #[derive(Facet)]
        #[facet(fg::namespace = "detail")]
        pub struct Pair(Option<Leaf>, Vec<Leaf>);

        #[derive(Facet)]
        #[facet(fg::namespace = "detail")]
        pub struct Wrapped(Vec<Option<Leaf>>);

        #[derive(Facet)]
        pub struct App {
            pub holder: Holder,
            pub payload: Payload,
            pub pair: Pair,
            pub wrapped: Wrapped,
            $(pub $leaf_in_root: Leaf,)?
        }
    };
}

#[test]
fn an_unannotated_type_is_referred_to_in_the_namespace_it_inherits() {
    holders!(;);

    let registry = reflect!(App).unwrap();
    assert_every_reference_names(
        &registry,
        &QualifiedTypeName::namespaced("detail".to_string(), "Leaf".to_string()),
        false,
    );
}

#[test]
fn a_type_pinned_to_root_is_referred_to_in_root() {
    holders!(#[facet(fg::namespace)]; leaf_in_root);

    let registry = reflect!(App).unwrap();
    assert_every_reference_names(
        &registry,
        &QualifiedTypeName::root("Leaf".to_string()),
        true,
    );
}

#[test]
fn an_explicitly_namespaced_type_is_referred_to_in_its_namespace() {
    holders!(#[facet(fg::namespace = "other")]; leaf_in_root);

    let registry = reflect!(App).unwrap();
    assert_every_reference_names(
        &registry,
        &QualifiedTypeName::namespaced("other".to_string(), "Leaf".to_string()),
        true,
    );
}

/// #167: a type with no namespace attribute, reached from a namespaced one, is referred to in
/// that namespace from inside `Option` as well as from inside `Vec`.
#[test]
fn an_inherited_namespace_is_kept_inside_an_option() {
    #[derive(Facet)]
    pub struct Neighbour {
        pub id: u32,
    }

    #[derive(Facet)]
    pub struct Neighbourhood {
        pub manager: Option<Neighbour>,
        pub reports: Vec<Neighbour>,
    }

    #[derive(Facet)]
    #[facet(fg::namespace = "detail")]
    pub struct ViewModel {
        pub neighbourhood: Neighbourhood,
    }

    #[derive(Facet)]
    pub struct App {
        pub detail: ViewModel,
    }

    insta::assert_yaml_snapshot!(reflect!(App).unwrap(), @"
    ? namespace: ROOT
      name: App
    : STRUCT:
        - - detail:
              - TYPENAME:
                  namespace:
                    NAMED: detail
                  name: ViewModel
              - []
        - []
    ? namespace:
        NAMED: detail
      name: Neighbour
    : STRUCT:
        - - id:
              - U32
              - []
        - []
    ? namespace:
        NAMED: detail
      name: Neighbourhood
    : STRUCT:
        - - manager:
              - OPTION:
                  TYPENAME:
                    namespace:
                      NAMED: detail
                    name: Neighbour
              - []
          - reports:
              - SEQ:
                  TYPENAME:
                    namespace:
                      NAMED: detail
                    name: Neighbour
              - []
        - []
    ? namespace:
        NAMED: detail
      name: ViewModel
    : STRUCT:
        - - neighbourhood:
              - TYPENAME:
                  namespace:
                    NAMED: detail
                  name: Neighbourhood
              - []
        - []
    ");
}

/// #160: a type pinned to ROOT keeps its pin when a namespaced type holds it inside a generic.
#[test]
fn a_root_pin_is_kept_inside_a_generic() {
    #[derive(Facet)]
    #[facet(fg::namespace)]
    pub struct Shared {
        pub id: u32,
    }

    #[derive(Facet)]
    #[facet(fg::namespace = "kv")]
    pub struct Entry {
        pub one: Shared,
        pub many: Vec<Option<Shared>>,
    }

    insta::assert_yaml_snapshot!(reflect!(Entry).unwrap(), @"
    ? namespace: ROOT
      name: Shared
    : STRUCT:
        - - id:
              - U32
              - []
        - []
    ? namespace:
        NAMED: kv
      name: Entry
    : STRUCT:
        - - one:
              - TYPENAME:
                  namespace: ROOT
                  name: Shared
              - []
          - many:
              - SEQ:
                  OPTION:
                    TYPENAME:
                      namespace: ROOT
                      name: Shared
              - []
        - []
    ");
}

/// An enum held by a newtype or tuple struct is the struct's field, rather than the enum's own
/// payloads leaking into the struct.
#[test]
fn an_enum_in_a_tuple_struct_is_its_field() {
    #[derive(Facet)]
    pub struct Leaf {
        pub id: u32,
    }

    #[derive(Facet)]
    #[repr(C)]
    #[allow(dead_code)]
    pub enum Choice {
        One(Leaf),
        Many(Vec<Leaf>),
    }

    #[derive(Facet)]
    pub struct Wrapper(Choice);

    #[derive(Facet)]
    pub struct Pair(u8, Choice);

    #[derive(Facet)]
    pub struct App {
        pub wrapper: Wrapper,
        pub pair: Pair,
    }

    let registry = reflect!(App).unwrap();
    let choice = Format::TypeName(QualifiedTypeName::root("Choice".to_string()));
    assert_eq!(
        registry[&QualifiedTypeName::root("Wrapper".to_string())],
        ContainerFormat::NewTypeStruct(Box::new(choice.clone()), Doc::default()),
    );
    assert_eq!(
        registry[&QualifiedTypeName::root("Pair".to_string())],
        ContainerFormat::TupleStruct(vec![Format::U8, choice], Doc::default()),
    );
}

#[test]
fn a_reference_to_an_unregistered_type_is_rejected() {
    let mut builder = RegistryBuilder::new();
    builder.registry.insert(
        QualifiedTypeName::namespaced("detail".to_string(), "Neighbourhood".to_string()),
        ContainerFormat::Struct(
            vec![Named {
                name: "manager".to_string(),
                doc: Doc::default(),
                value: Format::Option(Box::new(Format::TypeName(QualifiedTypeName::root(
                    "Neighbour".to_string(),
                )))),
            }],
            Doc::default(),
        ),
    );

    let err = builder.build().unwrap_err();
    assert_eq!(
        err,
        Error::DanglingTypeReference {
            name: "Neighbourhood".to_string(),
            namespace: "detail".to_string(),
            location: "manager".to_string(),
            missing_name: "Neighbour".to_string(),
            missing_namespace: "ROOT".to_string(),
        }
    );
    insta::assert_snapshot!(err, @r#"`manager` in "Neighbourhood" in namespace "detail" refers to "Neighbour" in namespace "ROOT", which is not a registered type. This is a bug in facet_generate's reflection; please report it"#);
}

#[test]
fn a_dangling_reference_in_an_enum_payload_is_located_by_its_variant() {
    let mut builder = RegistryBuilder::new();
    builder.registry.insert(
        QualifiedTypeName::root("Event".to_string()),
        ContainerFormat::Enum(
            BTreeMap::from([(
                0,
                Named {
                    name: "Moved".to_string(),
                    doc: Doc::default(),
                    value: VariantFormat::Struct(vec![Named {
                        name: "to".to_string(),
                        doc: Doc::default(),
                        value: Format::Seq(Box::new(Format::TypeName(QualifiedTypeName::root(
                            "Place".to_string(),
                        )))),
                    }]),
                },
            )]),
            crate::reflection::format::EnumTagging::External,
            Doc::default(),
        ),
    );

    let err = builder.build().unwrap_err();
    assert!(
        matches!(&err, Error::DanglingTypeReference { location, .. } if location == "Moved.to"),
        "{err:?}"
    );
}

/// facet gives `Range` `Def::Scalar`, although it is a user struct, so it is referred to as the
/// container it is registered as, rather than as a scalar (crux's `notes` example).
#[test]
fn a_range_in_an_enum_newtype_variant_is_a_container() {
    #[derive(Facet)]
    #[repr(C)]
    #[allow(dead_code)]
    pub enum Event {
        Selection(std::ops::Range<usize>),
    }

    insta::assert_yaml_snapshot!(reflect!(Event).unwrap(), @"
    ? namespace: ROOT
      name: Event
    : ENUM:
        - 0:
            Selection:
              - NEWTYPE:
                  TYPENAME:
                    namespace: ROOT
                    name: Range
              - []
        - EXTERNAL
        - []
    ? namespace: ROOT
      name: Range
    : STRUCT:
        - - start:
              - U64
              - []
          - end:
              - U64
              - []
        - []
    ");
}

/// The std types that facet exposes as user structs with `Def::Scalar` are containers from every
/// position a reference can be in, not only the direct ones.
#[test]
fn std_user_structs_with_a_scalar_def_are_containers_everywhere() {
    macro_rules! positions {
        ($($name:ident: $t:ty => $container:literal),* $(,)?) => {$(
            mod $name {
                #![allow(dead_code)]
                use super::*;

                #[derive(Facet)]
                pub struct Holder {
                    pub direct: $t,
                    pub option: Option<$t>,
                    pub seq: Vec<Option<$t>>,
                    pub map: BTreeMap<String, $t>,
                    pub tuple: ($t, u8),
                }

                #[derive(Facet)]
                pub struct NewType(pub $t);

                #[derive(Facet)]
                pub struct Pair(pub u8, pub Vec<$t>);

                #[derive(Facet)]
                #[repr(C)]
                pub enum Event {
                    NewType($t),
                    Option(Option<$t>),
                    Tuple(Vec<$t>, u8),
                    Struct { value: Option<$t> },
                }

                #[derive(Facet)]
                pub struct App {
                    pub holder: Holder,
                    pub new_type: NewType,
                    pub pair: Pair,
                    pub event: Event,
                }
            }

            {
                use $name::App;
                let registry =
                    reflect!(App).unwrap_or_else(|err| panic!("{}: {err}", stringify!($t)));
                let references = references_to(&registry, $container);
                assert_eq!(references.len(), 11, "{}: {references:#?}", stringify!($t));
                assert!(
                    registry.contains_key(&QualifiedTypeName::root($container.to_string())),
                    "{} is not registered",
                    $container,
                );
            }
        )*};
    }

    positions!(
        range: std::ops::Range<u32> => "Range",
        phantom: std::marker::PhantomData<u8> => "PhantomData",
        infallible: std::convert::Infallible => "Infallible",
    );
}

/// A type reflection doesn't support, such as `Result`, is skipped where it always was, rather
/// than panicking: here, as an element of a tuple struct or a tuple variant.
#[test]
fn an_unsupported_type_in_a_tuple_position_is_skipped() {
    #[derive(Facet)]
    pub struct Pair(pub u8, pub Result<u32, String>);

    #[derive(Facet)]
    #[repr(C)]
    #[allow(dead_code)]
    pub enum Event {
        Tuple(Result<u32, String>, u8),
    }

    let registry = reflect!(Pair, Event).unwrap();
    assert_eq!(
        registry[&QualifiedTypeName::root("Pair".to_string())],
        ContainerFormat::TupleStruct(vec![Format::U8], Doc::default()),
    );
    let ContainerFormat::Enum(variants, _, _) =
        &registry[&QualifiedTypeName::root("Event".to_string())]
    else {
        panic!("not an enum");
    };
    assert_eq!(variants[&0].value, VariantFormat::Tuple(vec![Format::U8]));
}

/// An optional payload of an unsupported scalar is an error, as it always was, rather than a
/// unit variant that silently drops the payload.
#[test]
fn an_optional_unsupported_scalar_payload_is_an_error() {
    #[derive(Facet)]
    #[repr(C)]
    #[allow(dead_code)]
    pub enum Event {
        Wait(Option<std::time::Duration>),
    }

    let err = reflect!(Event).unwrap_err();
    insta::assert_snapshot!(err.root_cause(), @r#"failed to add type Event: problem reflecting type '"Duration"': Scalar type is not supported and should be skipped"#);
}

/// An array in a tuple position is one element, not the array followed by its element type.
#[test]
fn an_array_in_a_tuple_position_is_one_element() {
    #[derive(Facet)]
    pub struct Leaf {
        pub id: u32,
    }

    #[derive(Facet)]
    pub struct Pair(pub u8, pub [Leaf; 2]);

    let registry = reflect!(Pair).unwrap();
    let leaves = Format::TupleArray {
        content: Box::new(Format::TypeName(QualifiedTypeName::root(
            "Leaf".to_string(),
        ))),
        size: 2,
    };
    assert_eq!(
        registry[&QualifiedTypeName::root("Pair".to_string())],
        ContainerFormat::TupleStruct(vec![Format::U8, leaves], Doc::default()),
    );
}
