Code for behaviour tracking anlaysis of experiments for paper "   "

An intial code for preprocessing the video using deeplabcut
-----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------

--Preprocessing folder: Here is the preprocessed data using deep lab cut ct [ref]. It is quite easy to use this.
Here the code in preprocessing_dlv.ioynb is the code used for generating the dlc filtered tables we will use 
for the analysis. The filtered dlc videos where generated using the dlc gui, it seems that there is a bug in the Corresponding dlc
function that does not recognize properly the video path, for this just run the following to run the GUI:

     python -m deeplabcut

In case you want to train the model from zero the raw data can be found in:
- https://saco.csic.es/apps/files/files/217962888?dir=/zic2/code_data_zic2/data/raw_data/prey_hunting

and the videos used for traning the model in our case:

- https://saco.csic.es/apps/files/files/222213335?dir=/zic2/code_data_zic2/data/raw_data/prey_hunting/videos_for_model_training

The labelled data and trained model used for the analysis be found in:

-  https://saco.csic.es/apps/files/files/222227744?dir=/zic2/code_data_zic2/data/raw_data/prey_hunting/dlc_preprocessed_data

------------------------------------------------------------------------------------------------------------------------
---------------------------------------------
---------------------------------------------


The analysis is divided in two main scripts:

1) A preprocessing script:
     The script preprocessing.ipynb  is used to preprocess the data of each experiment. The raw data is called from 
     folder of analysed videos and the sessions analysed are those indeicated in session.txt.
     Overall the code takes the raw matrix of position of each subject and clasiffies 3 types of behaviours:
          - Approach
          - Contact
          - Exploration

     The code loops through each session and computes in the anlysed folder:
     - analysed/tables:

          - behaviour_matrix_df: Agreggated behaviour matrix filtered by the behaviour_thr. 
                              behaviour_thr determines how long mut a behaviour last to be considered one of the three behaviours.
          - raw_matrix_df: Raw matrix dataframe with computed variables

     - analysed/video:
          - A labelled video for each session with the computed parameters and behavior type.
     ** More info in preprocessing.ipynb

2) Data mungling, statistics and figuras:
     The following notebooks are then used for figure creation, statistics and data mungling using the 
     prerocessed from the previous script.

     - analysis_by_contact.ipynb: Data analysis filtering approaches that end in contact.
     - analysis_by_time_thr.ipynb: Data analysis filtering behaviors by a time threshold- 
     - figures_approaches_contacts.ipynb: Example figures of approaches, explorations and contacts through each session.
     - figure_orientation_distance: Ditance to cricket and head orientation figures 
     - circular_statistics forlder: r code for figure 6 j-k.

Environments:
- analysis.yml: For all analysis
     conda env create -n analysis.yml

- tf.yml: Por deep lab cut preprocessing
     conda env create -n tf.yml



Author:                             Arturo José Valiño Pérez

Corresponding author:              Arturo José Valiño Pérez (arturo-jose.valino@incipit.csic.es)