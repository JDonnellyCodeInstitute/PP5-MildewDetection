# Ensure our src/ package is importable
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Streamlit configuration
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
