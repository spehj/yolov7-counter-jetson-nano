# Working Dockerfile for yolo-image container
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3 

COPY /requirements.txt requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 uninstall pillow -y
RUN pip3 install "pillow<9"
RUN pip3 install --no-cache-dir -r requirements.txt --ignore-installed && rm requirements.txt
RUN pip3 install imutils

RUN apt-get update && apt-get install -y python3-seaborn
RUN apt-get install -y python3-tk
# Seaborn error fix
# Remove the import of remove_na from seaborn's categorical.py
RUN sed -i '/from pandas.core.series import remove_na/d' /usr/lib/python3/dist-packages/seaborn/categorical.py

# Add the remove_na function right below the imports in categorical.py
RUN sed -i '/import pandas.core.series/a\
def remove_na(arr):\n\
    """\n\
    Helper method for removing NA values from array-like.\n\
    Parameters\n\
    ----------\n\
    arr : array-like\n\
        The array-like from which to remove NA values.\n\
    Returns\n\
    -------\n\
    clean_arr : array-like\n\
        The original array, just with NA values removed.\n\
    """\n\
    \n\
    return arr[pd.notnull(arr)]' /usr/lib/python3/dist-packages/seaborn/categorical.py
RUN pip3 install filterpy
WORKDIR /trt

CMD ["/bin/bash"]