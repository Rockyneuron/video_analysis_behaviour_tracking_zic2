Code for behaviour tracking analysis of experiments for paper [1].

-----------------------------------------------------------------------------------------------------------------------
Code for data analysis used in [1] 

----------------------------------------------------------------------------------------------------------------------

1) A preprocessing script:
     The script preprocessing.ipynb  is used to preprocess the data of each experiment. The raw data is called from the 
     folder "dlc_preprocessed_data" (deep lab cut preprocessed videos) that can be found in [4]. The sessions analysed are those indicated in session.txt.
     Overall the code takes the raw matrix of position labels of each subject and clasiffies 3 types of behaviours:
          - Approach
          - Contact
          - Exploration

     The code loops through each session and computes in the analysed folder:
     - analysed/tables:

          - behaviour_matrix_df: Agreggated behaviour matrix filtered by the behaviour_thr. 
          - raw_matrix_df: Raw matrix dataframe with computed variables

     - analysed/video:
          - A labelled video for each session with the computed parameters and behavior type.
     ** More info in preprocessing.ipynb

2) Data mungling, statistics and figures:
     The following notebooks are then used for figure creation, statistics and data mungling using the 
     prerocessed data from the previous script.

     - analysis_by_contact.ipynb: Data analysis filtering approaches that end in contact.
     - analysis_by_time_thr.ipynb: Data analysis filtering behaviors by a time threshold.
     - figures_approaches_contacts.ipynb: Example figures of approaches, explorations and contacts through each session.
     - figure_orientation_distance: Distance to cricket and head orientation figures 
     - circular_statistics folder: r code for figure 6 j-k.


-----------------------------------------------------------------------------------------------------------------------
Code for preprocessing the video using deeplabcut

----------------------------------------------------------------------------------------------------------------------

-preprocessing_deep_lab_cut:
Here the code in preprocessing_dlc.ipynb is the code used for generating the dlc filtered tables that were used
in the analysis. The filtered dlc videos where generated using the dlc gui. It seems that there is a bug in the corresponding dlc
function that does not recognize properly the video path. Therefore, just run the following to run the gui:

     python -m deeplabcut

In case you want to train the model from zero the raw data can be found in [2]

The videos used for traning the deep lab cut model in [3]:

The labelled data and trained model used for the analysis be found in [4]:
 
------------------------------------------------------------------------------------------------------------------------


Environments:
- analysis.yml: For all analysis
     conda env create -n analysis.yml

- tf.yml: Por deep lab cut preprocessing
     conda env create -n tf.yml

------------------------------------------------------------------------------------------------------------------------
Steps to run the analysis:

-----------------------------------------------------------------------------------------------------------------------


1) Install the analysis environment
2) Download the labelled data used for the analysis found in [4] and extract the folder dlc_preprocessed_data inside this repository. Delete the zip
(In case you have downloaded the cloud version of the code, you should already have the folder in the repository)
3) Run preprocessing.ipynb
4) Run the notebooks from step 2 (Data mungling, statistics and figures)


Author:                             Arturo José Valiño Pérez

Corresponding author:              Arturo José Valiño Pérez (arturo-jose.valino@incipit.csic.es)


1. Genetic Rewiring of Retinothalamic Neurons Induces Ocular Dominance Columns in mice and Enhances Binocular Vision and Predatory Behaviors 
2. https://saco.csic.es/apps/files/files/217962888?dir=/zic2/code_data_zic2/data/raw_data/prey_hunting
3. https://saco.csic.es/apps/files/files/222213335?dir=/zic2/code_data_zic2/data/raw_data/prey_hunting/videos_for_model_training
4. https://saco.csic.es/apps/files/files/222227744?dir=/zic2/code_data_zic2/data/raw_data/prey_hunting/dlc_preprocessed_data
