This folder contains training and testing datasets for Fig12.

Training dataset: Enhanced_RFMbest_train.csv

Testing dataset: (1) language.csv : language includes c, java and so on. (2) feature+threshold.csv: feature includes star, watcher, committer, community. threshold includes 0.01, 0.02, 0.04, 0.08, 0.15.


Features:
true_pdp: If the project is labeled as a public development project, this feature is set to TRUE. Otherwise,this feature is set to FALSE.

true_star: the star number of this projct.

true_watcher: the watcher number of this project.

true_committer: the committer number of this project.

true_community: the number of users that appears in the pull request, issues, and committer mumbers.

have_language: if this project has adopted a programming language, this feature is set to 1. Otherwise,this feature is set to 0.

is_null: if this project has no description this feature is set to 1. Otherwise,this feature is set to 0.

url_X: if the url of this project contains keyword X, this feature is set to 1. Otherwise,this feature is set to 0.

des_X: if the description of this project contains keyword X, this feature is set to 1. Otherwise,this feature is set to 0.