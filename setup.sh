mkdir -p ~/.doc_streamlit/                                               
echo "\                       
[server]\n\                       
port = $PORT\n\                       
enableCORS = false\n\                       
headless = true\n\                       
\n\                       
" > ~/.doc_streamlit/config.toml
