## This reposity holds the code for the paper Online Academic Course Performance Prediction using Relational Graph Convolutional Neural Network [DOPE](https://educationaldatamining.org/files/conferences/EDM2020/papers/paper_45.pdf) 

![DOPE](http://cse.msu.edu/~karimiha/images/dope.jpg)


## Instructions 
1. Make sure requirements are satisfied
`pip install -r requirements.txt`
2.Copy the project in a directory in your machine e.g., /home/XYZ/. Note that the dataset is in `Data/data.csv`

3. To run the experiment call `train.py`
    
    **Example** `python train.py --path /home/XYZ/ --experiment_name experiment1 --training_courses SS2,ST1 --testing_courses SS2,ST1 --training_periods 2013B,2013J --testing_periods 2014B,2014J`
    
    
    **Note 1.** Input courses and periods are seperated by comma (if more than one course/period is intended).
     
    **Note 2.** Please refer to `config.py` for different parameters.
     
## Citations

If you use the code in this repository, please cite the following paper


@inproceedings{karimi2020edm,
  title={Online Academic Course Performance Prediction using Relational Graph Convolutional Neural Network},
  author={Karimi, Hamid and Derr, Tyler and Huang, Jiangtao and Tang, Jiliang},
  booktitle={Proceedings of The 13th International Conference on Educational Data Mining (EDM 2020)},
  pages={444--460},
  year={2020}
}

## Contact
Web page: [http://cse.msu.edu/~karimiha/](http://cse.msu.edu/~karimiha/)

Email: [karimiha@msu.edu](karimiha@msu.edu)
