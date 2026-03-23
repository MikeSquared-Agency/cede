#[tokio::main]
async fn main() {
    if let Err(e) = cortex_embedded::cli::run().await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
