# Start API container and Streamlit UI
docker rm -f churn-api 2>$null
docker run -d --name churn-api -p 8010:8000 eznama/churn-api:v0.3
$env:STREAMLIT_BROWSER_GATHERUSAGESTATS="false"
python -m streamlit run .\streamlit_app.py --server.port 8501
