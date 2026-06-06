rm -rf /tmp/lanka_data \
    && python3 workflows/examples_build.py \
    && python3 -m pytest -v -p no:warnings "$@" \
    && python3 workflows/readme_build.py