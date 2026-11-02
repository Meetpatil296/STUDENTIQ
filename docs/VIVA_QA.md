# Viva Preparation - Student Performance Prediction System (Random Forest)

Answers are short and simple on purpose. Real project numbers: **1,600 unique students, 8 features, 3 classes, 80/20 split (1,280 / 320), 300 trees, test accuracy 85.31 %, weighted F1 85.29 %, 5-fold CV 84.92 %.**

---

### A. Artificial Intelligence and Machine Learning

**1. What is Artificial Intelligence?**
AI is the ability of a computer to perform tasks that normally need human intelligence, such as learning, reasoning, decision-making and understanding language.

**2. What is Machine Learning? How is it related to AI?**
Machine Learning is a part of AI in which a computer learns patterns from data instead of being given fixed rules. In this project the model learns from past student records how study habits relate to performance.

**3. What is supervised learning? Why is this project supervised?**
In supervised learning the training data contains both inputs and the correct answer (label). Here each student record has inputs (study hours, attendance, marks…) and the known label (Good / Average / Poor), so the model learns the mapping from inputs to label.

**4. What is classification? How is it different from regression?**
Classification predicts a *category* (Good / Average / Poor); regression predicts a *number* (for example 73.5 marks). Our output is a category, so this is a classification problem - specifically *multi-class* classification.

### B. Random Forest and Decision Trees

**5. What is a decision tree?**
A tree-shaped set of yes/no questions on feature values (e.g. "attendance > 80 %?") that ends in a prediction. It is easy to understand but a single deep tree can memorise the training data.

**6. What is a Random Forest?**
A collection of many decision trees (300 here) trained on different random samples; the final answer is the category chosen by most trees. It is an *ensemble* method.

**7. What are bootstrap sampling and majority voting?**
*Bootstrap sampling:* each tree is trained on a random sample of students taken **with replacement**, so every tree sees slightly different data. *Majority voting:* every tree predicts a class, and the class with the most votes is the final prediction.

**8. Why did you choose Random Forest?**
It works very well on small tabular data, handles non-linear patterns (like the "best sleep is 7-8 hours" effect), is robust to noise, needs no feature scaling, handles three classes naturally, reduces overfitting compared with one tree and shows feature importance.

**9. What is Gini impurity?**
A measure of how mixed the classes are in a node. A pure node (all one class) has Gini 0. The tree chooses the split that reduces impurity the most.

**10. Name the important hyper-parameters you used.**
`n_estimators=300` (trees), `max_depth=9` (limits tree size), `min_samples_split=10`, `min_samples_leaf=5`, `max_features="sqrt"` (random features per split), `class_weight="balanced"`, `random_state=42` (reproducibility).

**11. What is feature importance? Which features were most important?**
It shows how much each feature helped the trees to split correctly. Most important: previous exam marks (31.4 %), internal marks (21.9 %), assignments (19.0 %), attendance (13.6 %), study hours (10.7 %). Internet availability and extracurricular activity were the least important.

### C. Data, Preprocessing and Libraries

**12. What is a dataset? Where does your data come from?**
A dataset is an organised collection of data used for learning. Ours is **synthetically generated** by `generate_dataset.py` - it is not real student data - and is stored as a structured CSV file (not a database). It has 1,600 unique students plus 20 duplicates (1,620 rows).

**13. Why did you generate synthetic data, and how did you make it realistic?**
Real student records are private and hard to obtain. We simulated hidden traits (aptitude, diligence), generated features from them with random noise, computed a weighted performance score with extra noise, and split it into categories by percentiles. So features truly relate to the label, but categories still overlap - labels are not random.

**14. What is data preprocessing? Which steps did you perform?**
Preparing raw data for the model: loading, checking missing values, removing duplicates (20 removed), validating values, imputing missing values (median for numbers, most frequent for Yes/No), one-hot encoding, separating features and target, and a stratified train-test split.

**15. What is data leakage and how did you avoid it?**
Leakage is when information from the test data influences training, giving unrealistically good results. We avoided it by removing duplicates before the split (no copy in both sets), putting the imputer and encoder inside a `Pipeline` that is fitted on training data only, and excluding Student_ID from the features.

**16. Why is feature scaling not needed?**
Decision trees split on thresholds (e.g. "marks > 65"), so changing the scale of a feature does not change the splits. Scaling matters for distance-based or gradient-based models, not for Random Forest.

**17. What is stratified splitting?**
Splitting so that the class proportions (Good / Average / Poor) are the same in the training and test sets. Our test set has 31.9 % Good, 38.1 % Average, 30.0 % Poor - almost identical to the training set.

**18. What are Pandas, NumPy and Scikit-learn used for?**
*Pandas:* tables (DataFrames), cleaning and analysis. *NumPy:* fast numerical arrays and random numbers (used in data generation). *Scikit-learn:* machine-learning tools - pipeline, imputer, encoder, Random Forest, train-test split and metrics.

**19. What is a Pipeline and why is it useful?**
A Pipeline chains preprocessing and the model into one object. It guarantees that exactly the same steps run in training, testing and in the app, and prevents leakage.

### D. Training, Testing and Evaluation

**20. Why do we divide data into training and testing sets?**
To check whether the model works on *new* students. The model learns from the training data (80 %) and is evaluated on test data (20 %) it has never seen.

**21. What is overfitting? What is underfitting? How did you control them?**
*Overfitting:* the model memorises training data and performs badly on new data (high training accuracy, low test accuracy). *Underfitting:* the model is too simple and performs badly on both. We controlled overfitting with many trees, limited depth, minimum samples per leaf and cross-validation. Training accuracy is 93.5 % and test accuracy 85.3 % - a moderate, acceptable gap.

**22. What is accuracy? Is it always enough?**
Accuracy = correct predictions ÷ total predictions (273 / 320 = 85.31 %). It is reliable only when classes are fairly balanced, as in our dataset (30 / 38 / 32 %); otherwise precision, recall and F1 are also needed.

**23. Explain precision, recall and F1-score.**
*Precision:* of the students predicted "Good", how many truly are Good. *Recall:* of the truly Good students, how many the model found. *F1-score:* the harmonic mean of the two. Example (Poor class): precision 0.857, recall 0.938, F1 0.896.

**24. What is a confusion matrix? Explain yours.**
A table of actual versus predicted classes. The diagonal (83, 100, 90) shows correct predictions. Errors occur only between neighbouring categories (e.g. 19 Good students predicted as Average); no Good student was predicted Poor and no Poor student was predicted Good.

**25. What is cross-validation and why did you use it?**
Training data is split into 5 parts; the model trains on 4 and validates on 1, rotating 5 times. The mean accuracy (84.92 % ± 1.57 %) shows that the 85.31 % test accuracy is not just luck of one split.

**26. Why is your accuracy about 85 % and not 100 %?**
Real academic performance is influenced by many things beyond the eight features, and our data deliberately includes noise and overlap between categories. A near-100 % result would suggest leakage or an unrealistically easy dataset.

**27. What is the out-of-bag (OOB) score?**
Each tree does not see about 37 % of the students in its bootstrap sample. Predicting those students with the trees that did not see them gives a free validation estimate (86.41 %).

### E. Prediction System and Streamlit

**28. What is Streamlit? Why did you use it?**
Streamlit is a Python library that turns scripts into interactive web apps without HTML/CSS/JavaScript. It is quick to build, easy to demonstrate and keeps the whole project in Python.

**29. How does the app produce a prediction? Is anything hard-coded?**
Nothing is hard-coded. The input is validated, converted into a one-row table, passed through the saved pipeline (imputation + encoding) and the saved Random Forest's trees vote; the vote share gives the probabilities. The model is loaded from `random_forest_model.pkl` with Joblib.

**30. How does the app handle invalid or missing inputs?**
Number fields have minimum and maximum limits; additional checks reject impossible values (e.g. study + sleep > 24 h) and warn about values outside the training range. A user may mark a field as "unknown": it becomes `NaN` and is filled with the training median / most frequent value by the pipeline.

**31. How do you explain a single prediction?**
We replace each input with the value of a typical student and measure how much the predicted category's probability changes. A positive change means the feature supported the prediction. This explanation comes from the model itself.

### Bonus questions

**32. What are the limitations of your project?** The data is synthetic, only eight features are used, three broad classes, and predictions are probabilities that should support, not replace, a teacher's judgement.

**33. How can the project be improved?** Use real anonymised data, add more features, tune hyper-parameters with GridSearchCV, add SHAP explanations, batch prediction from CSV and deployment online.

**34. Why is the model saved with Joblib?** Training takes time; saving lets the app load the trained model instantly. Joblib is efficient for scikit-learn objects containing large NumPy arrays.

**35. Is it ethical to predict student performance?** It is acceptable when used to *support* students, with privacy protection and human oversight; it must not be used to label, rank or punish students.
