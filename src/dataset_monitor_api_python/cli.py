import uvicorn

def main():
    uvicorn.run(
        "dataset_monitor_api_python.main:app",
        host="0.0.0.0",
        port=3000
    )

if __name__ == "__main__":
    main()