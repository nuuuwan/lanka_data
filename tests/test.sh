git add src
python3 workflows/single.py "LK" \
&& python3 workflows/single.py "LK-71:district/Map" \
&& python3 workflows/single.py "LK-pre1959:district/Religion/2012/Map" \
&& python3 workflows/examples_build.py \
    && python3 -m pytest -v -p no:warnings "$@" \
    && python3 workflows/readme_build.py