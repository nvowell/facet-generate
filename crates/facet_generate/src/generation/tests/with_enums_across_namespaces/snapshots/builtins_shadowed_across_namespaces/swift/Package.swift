// swift-tools-version: 5.8
import PackageDescription

let package = Package(
    name: "Example",
    products: [
        .library(
            name: "Example",
            targets: ["Example", "Kit", "Serde"]
        )
    ],
    targets: [
        .target(
            name: "Example",
            dependencies: ["Kit", "Serde"]
        ),
        .target(
            name: "Kit",
            dependencies: ["Serde"]
        ),
        .target(
            name: "Serde",
            dependencies: []
        ),
    ]
)
