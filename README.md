# NYC_Taxi_Saver

# Download dataset
The dataset can be found here for download:
ttps://drive.google.com/drive/folders/1Ro5BSQHwwPZ8oBOcDKWLmqG9vKUsxERV?usp=sharing

# Code Organization
<pre>

data/
-- __init__.py - data init
-- nyctaxi_dataset.py - Helper function to preprocess dataset
-- dataset.py - Preprocess data and get it in the necessary format

utils/
-- __init__.py - utils init
-- array_tool.py - Tools to convert specified type
-- config.py - Settings to configure the model 
-- constants.py - Declared constants
-- eval_tool.py - Tools to evaluate the accuracy of our model
-- logger.py - Logging module
-- geospatial_tools.py - Functions to conduct geospatial analysis
-- stats_tools.py - Tools to compute statistical methods
-- vis_tool.py - Tools to help visualize 

main.ipynb - Main notebook for conducting dataset visualization and statistics
linear_regression.py - Train and evaluate linear regression model

</pre>
