//! Generates the gRPC agent API (`proto/zai/v1/agent.proto`) with a vendored `protoc`, so no
//! image needs a system protobuf compiler.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    println!("cargo:rerun-if-changed=../../proto");
    tonic_prost_build::configure()
        .compile_protos(&["../../proto/zai/v1/agent.proto"], &["../../proto"])?;
    Ok(())
}
