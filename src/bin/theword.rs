#[tokio::main]
async fn main() {
    if let Err(e) = theword::cli::run().await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
