# Fake_review
---------------------------------------------------
Monitoring Fake Reviews Using RNN and Bi-LSTM
Classification Models
----------------------------------------------------------
Abstract
--------------------
The proposed, implemented and evaluvated a fake review monitoring system based on supervised
machine learning and deep learning methods on a review yelp-based dataset. The study touched
upon the issue of fake online reviews that may mislead consumers, tarnish the reputation of the
business and decrease the trust in online services/platforms. The application was developed in
Google Colab using Python and a quantitative experimental approach was used. The data was
loaded, checked and cleaned and then modelled. The review text and label columns were identified
with the help of the code, which then eliminated the unusable records, cleaned text features were
prepared, and a supervised classification dataset was created. Exploratory data analysis was
applied to investigate missing values, distribution of classes, rating distribution number of words
and other properties of review-text. TF-IDF features were used to develop traditional machine
learning models such as Logistic Regression, Naive Bayes and Linear SVM. Moreover, RNN and
Bi-LSTM models were trained on tokenised and padded sequence of reviews. Accuracy, precision,
recall, F1-score, ROC-AUC, comparison of confusion matrices and ROC curve were used to
measure model performance. The findings indicated that the models performed moderately with
the highest visible performance of about 67-69%. The Bi-LSTM was found to be outperformed
the basic RNN and was chosen to be the final deep learning model,.The research finds that review-
text classification, can be used to aid fake review detection, though better accuracy is achieved
with behavioural features, better tuning and external validation. The future work will involve
reviewer activity, business level patterns, burst detection and transformer-based models to enhance
practical reliability and platform level monitoring effectiveness in general.
Keywords: Fake review detection, Yelp dataset, machine learning, deep learning, RNN, Bi-
LSTM, TF-IDF, text classification, online reviews.


Table of Contents
--------------------------------------------------------------------
Chapter 1: Introduction ................................................................................................................... 8
1.1 Background ........................................................................................................................... 8
1.2 Research Problem ................................................................................................................. 9
1.3 Rationale of the Study ........................................................................................................... 9
1.4 Research Aim ...................................................................................................................... 10
1.5 Research Objectives ............................................................................................................ 10
1.6 Research Questions ............................................................................................................. 10
1.7 Study Scope ........................................................................................................................ 10
1.8 Importance of the study .......................................................................................................11
1.9 Expected Outcome ...............................................................................................................11
1.10 Chapter Summary ..............................................................................................................11
Chapter 2: Literature Review ........................................................................................................ 12
2.1 Introduction ......................................................................................................................... 12
2.2 Fake Review Problem and Online Reviews........................................................................ 12
2.3 Linguistic and Traditional Machine Learning Methods...................................................... 13
2.4 Deep Learning and Sequential Text Modelling .................................................................. 13
2.5 Contextual Understanding and Bi-LSTM ........................................................................... 14
2.6 Attention, Position Information and Complexity of Models .............................................. 14
2.7 Behavioural, Network and Group-Based Detection ........................................................... 15
2.8 Yelp Data and Suitability of Data ....................................................................................... 15
5
2.9 Metrics and Claims of Performance ................................................................................... 16
2.10 Synthesis and Research Gap ............................................................................................. 16
2.11 Implications to the current study ....................................................................................... 17
2.12 Practical and Ethical Concerns ......................................................................................... 18
2.13 Chapter Summary ............................................................................................................. 18
Chapter 3: Research Methodology................................................................................................ 20
3.1 Introduction ......................................................................................................................... 20
3.2 Research Design.................................................................................................................. 20
3.4 Tools and Technologies ....................................................................................................... 20
3.4 Sampling and Data Source .................................................................................................. 21
3.5 Data Cleaning and Preparation ........................................................................................... 21
3.6 Feature Engineering ............................................................................................................ 22
3.7 Model Development............................................................................................................ 22
3.8 Evaluation Strategy ............................................................................................................. 22
3.9 Artefact Output.................................................................................................................... 23
3.10 Ethical and methodological limitations ............................................................................ 23
3.11 Chapter Summary ............................................................................................................. 23
Chapter 4: Results and Discussion ................................................................................................ 24
4.1 Introduction ......................................................................................................................... 24
4.2 Dataset Loading and Structure ............................................................................................ 24
4.3 Data Cleaning Results ......................................................................................................... 25
4.4 Exploratory Data Analysis Results ..................................................................................... 26
4.5 Model Preparation Results .................................................................................................. 28
4.6 Classic Machine Learning Performance ............................................................................. 28
4.7 RNN Results ....................................................................................................................... 29
6
4.8 Bi-LSTM Results ................................................................................................................ 30
4.9 Comparison to the expectations .......................................................................................... 31
4.10 Discussion ......................................................................................................................... 32
4.11 Practical Interpretation ...................................................................................................... 33
4.12 Chapter Summary ............................................................................................................. 33
Chapter 5: Conclusion and Recommendation ............................................................................... 35
5.1 Introduction ......................................................................................................................... 35
5.2 Overall Conclusion ............................................................................................................. 35
5.3 Achievement of Objectives ................................................................................................. 35
5.4 Key Findings ....................................................................................................................... 36
5.5 Contribution ........................................................................................................................ 36
5.6 Limitations .......................................................................................................................... 37
5.7 Recommendations ............................................................................................................... 37
5.8 Future work ......................................................................................................................... 38
5.9 Final Summary .................................................................................................................... 38
Appendix A: Supervisor Meeting / Monitoring Progress Report ............................................. 39
Appendix B: Sample Dataset Records ...................................................................................... 40
Appendix C: Code Repository / Notebook Access ................................................................... 40
Appendix D: Extended Results Tables...................................................................................... 40
Appendix E: Glossary of Terms and Fiqures of each results ................................................... 41
References ..................................................................................................................................... 44

List of Figures

1.Data Cleaning Results --------------------------------------------------25
2.Target Distribution ------------------------------------------------------26
3.Rating Distribution ------------------------------------------------------27
4.Word Count Distribution -----------------------------------------------27
5.Treditional Machine Learning -----------------------------------------29


Chapter 1: Introduction
-------------------------------------------------------
1.1 Background

Online reviews have also become an important part of digital purchasing behaviour since
consumers base their decisions on these reviews to determine the quality of products, the reliability
of the services, and the reputation of a business. However, fake, spam or deceptive reviews can be
posted on the same review systems, diminishing trust in the online platforms. False reviews are
hard to detect as they may have a similar tone, length, and sentiment of the real customer reviews.
According to Chen et al. (2025), fake reviews impact consumer choice and establish a necessity to
find effective detection techniques. This paper thus aims to track fake reviews with the help of text
classification with Recurrent Neural Network (RNN) and Bidirectional Long Short-Term Memory
(Bi-LSTM) models.

1.2 Research Problem

The research problem is that online sites can have high amounts of review text as well, and manual
monitoring is not practical. Conventional machine learning systems are able to classify text,
however, they usually rely on human-crafted features and they might be insufficient to learn the
contextual meaning of the sentences in the review. Kennedy et al. (2020) demonstrate that it is
significant that contextual representation is considered in opinion spam detection since the
meaning of reviews cannot always be determined based on separated words. This poses an issue
of fake review monitoring since fake reviews can be written in natural language, emotional words
and realistic information so they cannot be detected. So one needs a model that can be trained to
acquire sequential language patterns using review text and be able to aid in automatic classification
of fake and genuine reviews.

1.3 Rationale of the Study

The RNN and Bi-LSTM models can be used in this issue since online reviews consist of a sequence
of words. The RNN is capable of handling text sequentially, but it might have a hard time handling
longer reviews or reviews with significant meaning towards the end of a sentence. Bi-LSTM
enhances this by reading text being read both forward and backward to enable the model to
comprehend a broader context of a sentence. Mohawesh et al. (2021) believe that the use of models
that extract more profound semantic patterns as opposed to models that make use of surface-level
linguistic features is advantageous in terms of fake review detection. The paper also has the
traditional machine learning baselines to ensure that the results of RNN and Bi-LSTM can be fairly
compared.

1.4 Research Aim

This study will seek to design, implement and assess a fake review monitoring model based on
RNN and Bi-LSTM classification methods on a review dataset based on Yelp.

1.5 Research Objectives

The first will be to overview of the recent literature on fake review detection with focus on machine
learning, deep learning and sequence-based text classification. The second point will be to prepare
and clean Yelp data to form fake review classification. The third goal will be to apply baseline
machine learning models with TF-IDF features. The fourth goal will be to create RNN and Bi-
LSTM models of review classification. The fifth goal is to make comparisons between model
performance in terms of accuracy, precision, recall, F1-score and ROC-AUC. The sixth goal will
be to debate the merits and drawbacks of the created fake review monitoring strategy.

1.6 Research Questions

RQ1: How can a Yelp review dataset based on fake reviews be cleaned and ready to be classified?
RQ2: To what extent can both fake and genuine reviews be categorized using text features by
traditional machine learning models?
RQ3: To what extent can RNN and Bi-LSTM models classify fake and genuine reviews based on
sequential texts of reviews?
RQ4: Which model has the best performance in terms of accuracy, precision, recall, F1-score and
ROC-AUC?

1.7 Study Scope

This paper will only cover fake review monitoring by text classification. It is implemented on text
reviews, cleaned text features, baseline machine learning models, RNN and Bi-LSTM. The
research does not purport to eliminate false reviews on legitimate platforms and eliminate human
moderation. Neither does it involve burst detection as an essential implementation since the latter
needs timestamps that are reliable, reviewer identities and business-level activity patterns. He et
al. (2024) demonstrate that the network and behavioural characteristics are capable of assisting in
detecting fake reviews, yet the behavioural analysis is perceived as an addition in the future instead
of a key necessity in this study. This makes the project realistic, focused and in line with the
available resources.

1.8 Importance of the study

The problem of fake reviews is important since it might hurt the level of customer confidence,
distort buying behavior and influence fair competition among companies. A classification model
that works can also aid in the early detection since it will show suspicious text in reviews which
can be detected further. The information systems view of it is useful since data preprocessing,
natural language processing, supervised learning and model evaluation are all in a single applied
artefact. The ultimate implementation gives evidence in terms of whether Bi-LSTM has a better
classification of fake reviews in comparison to simple models. It also shows the shortcomings of
automated detection and the necessity to carefully interpret model outputs.

1.9 Expected Outcome
The anticipated results of this research are that a fake review monitoring model will be developed
that will be able to categorize Yelp reviews as fake or genuine with an expected accuracy range of
about 80-82. This percentage implies that the system will be approximately 80 to 82 reviews out
of 100 reviews in the testing dataset in the planned modelling approach. This percentage is
however, considered to be an anticipated standard and not a sure outcome since the ultimate
accuracy attained will be reliant on the dataset quality, class balance, text features and performance
of the model once it has been trained. The paper thus appraises the Logistic Regression, Naive
Bayes, Linear SVM, RNN and Bi-LSTM models to determine whether the end model achieves or
exceeds this expected range of accuracy.

1.10 Chapter Summary
In this chapter, the background, research problem, research rationale, aim, objectives, research
questions, scope and significance of the research were presented. The second chapter is a review
of the recent literature on fake review detection, and describes the technical basis of using RNN
and Bi-LSTM models.


Chapter 2: Literature Review
----------------------------------------------------------------------------------------------
2.1 Introduction

This chapter critically analyzes the recent literature related to fake review detection with a special
focus on text classification, RNN and Bi-LSTM models, Yelp-based datasets, and the practices of
evaluation. The analysis is organized around the shift to the deep learning approaches, as opposed
to the traditional features-based detection. It also mentions the drawbacks of text-only detection
since fake review behaviour may consist of manipulation of language in addition to platform-level
activity. Chen et al. (2025) state that detection of fake reviews is a problem as deceptive reviews
are not homogeneous but are subtle and hard to be reflected by the traditional sequential models.
Nonetheless, it does not imply that complex models are necessarily better when it comes to applied
projects. He et al. (2022) demonstrate that research in spam review detection comprises of
statistical methods, machine learning methods, deep learning and network-based methods which
implies that the choice of model must be determined by the structure of data sets and the aim of
the project. Thus, the current paper is concerned with a realistic implementation with the support
of the Yelp review text, traditional baselines, RNN and Bi-LSTM, and behavioural burst detection
is an improvement that can be considered in the future.

2.2 Fake Review Problem and Online Reviews

Customer trust is affected by online reviews since it is a social proof used in online markets.
According to Abri et al. (2020), customers often peruse past reviews prior to buying goods or
services, and it is commercially appealing to manipulate the reviews. This opinion is corroborated
by Chen et al. (2025), who posit that fake reviews hurt the consumer decision making, business
reputation and e-commerce order. But not all such suspicious reviews are handled equally in the
literature. Kumar (2020) makes a distinction between untruthful reviews, non-review and brand-
only reviews, implying that the concept of fake review detection is not as limited as positive or
negative sentiment analysis. This is a significant difference since a favorable review does not
necessarily mean it is fake, and a negative review does not necessarily mean that it is authentic.
Soldner et al. (2021) caution that certain fake review classifiers might acquire unwanted dataset
differences, instead of actual deception. This poses a critical issue to research design: high
accuracy can be very impressive but it can be unrealistic of actual fake review knowledge. This is
why this dissertation considers fake review monitoring as a classification aiding system but not an
ultimate moderation authority.

2.3 Linguistic and Traditional Machine Learning Methods

The common fake review detection algorithms are more traditional and start with linguistic and
machine learning classifiers. Abri et al. (2020) determined that redundancy, the length and the style
of language used in the reviews are some of the features that can be used to differentiate between
fake and authentic reviews. This justifies text preprocessing, tokenisation and TF-IDF
representation of an applied pipeline. But, there are obvious weaknesses in linguistic features.
Kennedy et al. (2020) claim that the contextual representation is helpful in detecting opinion spam
due to missing the underlying meaning with the help of word-level and handcrafted
representations. Gutierrez-Espinoza et al. (2020) discovered that ensemble learning is able to
perform better than certain traditional classifiers, indicating that traditional models can still be
helpful, as long as they are carefully designed. However, this enhancement fails to eliminate the
primary drawback: most models like Logistic Regression, Naive Bayes and SVM tend to treat text
as a weighted term instead of ranked language. He et al. (2022) also mention that most of the
previous techniques are based on feature extraction, although spam authors have the capability to
modify the wording of reviews and decreasing the robustness of these features. Thus, in this study,
it is best to use traditional models as a baseline, rather than the ultimate presumed solution.

2.4 Deep Learning and Sequential Text Modelling

Deep learning was appealing due to the fact that reviews are not a collection of isolated words, but
they are sequences. The comparison of traditional models with a deep learning model (MLP, CNN
and LSTM) as spam review detection is made by Shahariar et al. (2022), indicating that the neural
version can be evaluated with a classical one. RNN models are applicable since they process the
tokens sequentially and have the ability to have some sequence dependency. But longer reviews
can be a problem since previous information can be undermined when being processed by RNNs.
According to Chen et al. (2025), the conventional sequence models are challenged by the fact that
they are not able to capture the positional information and can be prone to losing vital information
in lengthy texts. This is in favor of LSTM-based models, yet it points to a practical problem. In the
case when dataset includes brief and noisy reviews, a complicated deep model is likely to overfit
rather than transfer learns deception features. Salunkhe (2021) advocates attention-based BiLSTM
to deceptive opinion spam due to its enhancement of the semantic representation, but this does not
imply that attention needs to be applied to all MSc artefacts. A less complex RNN and Bi-LSTM
comparison is still warranted since it will determine whether bidirectional sequence learning can
be used to enhance the Yelp classification task.

2.5 Contextual Understanding and Bi-LSTM

Bi-LSTM is significant as it reads the text forward and backward. According to Chen et al. (2025),
BiLSTM is capable of capturing past and future dependencies, whereas LSTM is primarily
involved with previous context. This is most applicable to fake review monitoring since use of
deceptive words can be based on the way the arguments are made throughout the entire sentence.
Mohawesh et al. (2021) state that linguistic-only based models are not be able to capture semantic
meaning, which contributes to more in-depth contextual modelling. Nonetheless, Bi-LSTM cannot
be considered as objectively better in all instances. The potential of contextualised embeddings
demonstrated by Kennedy et al. (2020) suggests that the deep meaning can also be represented by
transformer-based models. Conversely, transformers are not the primary technique in this
dissertation due to the approved topic of RNN and Bi-LSTM, and the complexity of a more
complex architecture is less straightforward to explain, train and test. The results of Chen et al.
(2025) are good, but they used the PosAtt-BiLSTM with positional encoding and hybrid attention,
thereby being more complicated than the suggested implementation. Thus, the current research
modifies the fundamental concept of Bi-LSTM logic without asserting to reproduce the original
bottom paper.

2.6 Attention, Position Information and Complexity of Models

Literature on the topic of sequence models suggests that attention and position-sensitive
representations are advantageous to sequence models. To enhance the detection of fake reviews,
Chen et al. (2025) suggest PosAtt-BiLSTM, which is a non-negative sinusoidal positional
encoding with hybrid attention to enhance fake review detection. Their findings are applicable
since ablation revealed that either taking away attention or BiLSTM decreased performance.
Nonetheless, these results need to be treated with caution during an MSc implementation. Complex
architecture can enhance benchmark performance, but comes with higher training time,
hyperparameter sensitivity and explanation. Salunkhe (2021) also agrees with attention-based
BiLSTM on deceptive review classification, indicating that the attention can be used to focus on
the significant review terms. Attention is no replacement of good dataset design though. Soldner
et al. (2021) show that the confounds of data-origin and product-ownership can exaggerate
classification performance, that is, architectural sophistication cannot correct biased data. Such a
contradiction is the main focus of this project. The model can be enhanced over time, but the
primary need is to establish a clear comparison between the baseline models, RNN and Bi-LSTM
on the same dataset based on Yelp.

2.7 Behavioural, Network and Group-Based Detection

One of the weaknesses of text-only detection is that the bogus reviews can be linguistically
persuasive. Kumar (2020) states that features like deviation in reviewers and review similarity are
considered to hold significance in terms of behavioural and contextual features. This favors the
notion that fake review detection is not just a natural language processing issue. According to Cao
et al. (2021), spam (activity) may manifest itself within groups, as groups of reviewers manipulate
the product reputation together. This puts a strain on individual-review classification as one review
can appear genuine and the reviewer group suspicious. He et al. (2024) reinforce this idea by
demonstrating that text or basic metadata can be more easily manipulated than features based on
networks. Nonetheless, these methods need user IDs, product IDs, time or reviewer networks
which might not be present in all data sets. Alam et al. (2021) demonstrate that Yelp data may
include reviews, users, business, and check-in data. However, the version of the SciDB should be
verified before making any claim. Thus, burst detection and reviewer reputation scoring remain as
future improvements in this dissertation unless supported by the final dataset.

2.8 Yelp Data and Suitability of Data

One of the most critical problems of the fake review research is the suitability of the dataset.
According to Alam et al. (2021), the Yelp data can be helpful to analyse reviews, businesses, users
and time trends. Yelp is an appropriate point of origin for a realistic fake review monitoring
artefact. But it is contingent on whether the downloaded dataset contains both text of the reviews
and an available fake or actual label that can be used. Chen et al. (2025) utilized Spam, Yelp Hotel
and Yelp Restaurant datasets and revealed that Yelp-based review information is welcomed in fake
review detection studies. However, using such datasets does not necessarily imply that all versions
of Yelp datasets can be used to classify fake reviews using supervision. Soldner et al. (2021) warn
against the fact that classifiers can use the source of the data instead of deception, which is
particularly important when datasets are merged or relabelled in an irresponsible way. This is the
reason why the present dissertation relies on a single Yelp dataset, as recommended by the
supervisor and does not merge unsupported datasets. Burst detection should be methodologically
inappropriate in a dataset that does not have fields like timestamp or reviewer-history and should
be included in future research.

2.9 Metrics and Claims of Performance

Fake review detection Model evaluation should have accuracy as well as be fake. Accuracy and
F1-score are used by Chen et al. (2025), and they justify that F1-score balances the precision and
recall. This is significant since there are two types of risks associated with a fake review detection
system: false negatives (not identifying fake reviews) and false positives (identifying a real
review). Accuracy is important due to the fact that false accusations might be detrimental to honest
reviewers, whereas recall is relevant since fake reviews that go unnoticed will be perpetually
misleading users. Kennedy et al. (2020) demonstrate that it is helpful to make comparisons among
several techniques, but only in the case of the same train-test split and preprocessing logic.
Shahariar et al. (2022) compare the traditional and deep learning techniques as well, which justify
the choice of the dissertation to involve a baseline model and a neural model. Performance however
should not be unrealistic as reported. According to Chen et al. (2025), the accuracy of an advanced
PosAtt-BiLSTM architecture on the Spam dataset is 88.75% and therefore a simple RNN/Bi-
LSTM project cannot claim to be over 90% accurate until it is tested. Results in the current study
are thus assessed empirically and accuracy, precision, recall, F1-score and ROC-AUC are reported.

2.10 Synthesis and Research Gap

Three patterns of the review of literature are observed. First, conventional machine learning could
still be useful in terms of baselines since it is interpretable and efficient. Gutierrez-Espinoza et al.
(2020) are able to show that ensemble and conventional models can also work well in the case of
informative text features. Second, deep learning is more robust where the sequence and contextual
meaning is important. Mohawesh et al. (2021) argue in favour of contextual modelling since fake
and genuine reviews may vary on the aspect of hidden semantic patterns. Third, behavioural and
network approaches are potential yet data-related. He et al. (2024) demonstrate that manipulation
that text models can overlook can be identified in network structure, but richer relational data is
needed (as compared to simple review text). This dissertation gap is practical as opposed to being
purely theoretical. Most studies suggest more complex architectures or higher-level behavioural
systems, yet an MSc artefact must have an implementation that is focused, reproducible and
explainable. It is this gap that is pertinent to the current study as the use of multiple datasets could
introduce complexity in terms of data structures, preprocessing requirements, labelling schemes,
and consistency of evaluation. Therefore, using one dataset enhances consistency throughout the
data preparation, modelling and evaluation process if not it makes more difficult to test all the
more advanced features as presented in the literature. Soldner et al. (2021) reveal the rationale
behind this warning, since an irresponsible dataset is designed in a way that can mislead the user
to believe that the model is more effective than it is. The dissertation thus focuses on the internal
consistency rather than expansion of features. This is not as ambitious as a full multi-signal system,
but is more robust to a reproducible MSc implementation. This keeps within the scope of the
research and scope of implementation.
This work fills that gap through the use of one dataset based on Yelp, with consistent cleaning and
preprocessing, pitting traditional baselines and RNN and Bi-LSTM models, and discussing the
burst detection as a future improvement and enhancement when appropriate data is present.

2.11 Implications to the current study

This dissertation has a direct influence on the practical design, which is informed by the literature.
The initial implication is that cleaning and preprocessing are not a normal technical procedure but
a model validity. He et al. (2022) underline that spam review research relies on datasets, types and
methods of features detection, which implies that ambiguous preprocessing may impact
subsequent claims. This aids in the notebook format, which verifies missing values, duplication,
length of text and class ratio followed by modelling. The second implication is that there is still a
need of baseline models. When the Bi-LSTM is contrasted with just RNN, the analysis will not be
able to demonstrate whether sequence learning will be of any benefit over the more basic TF-IDF
techniques. Gutierrez-Espinoza et al. (2020) demonstrate that conventional and ensemble models
can be very strong and therefore, omitting them would make a weak comparison.
The third implication has to do with realism. Chen et al. (2025) demonstrate that more complex
models are capable of good results, however, their model contains elements that are not a typical
Bi-LSTM. This dissertation should thus not purport that a smaller implementation will be bound
to perform better than the base paper. The fourth implication is related to data limits. Cao et al.
(2021) and He et al. (2024) demonstrate that fake review manipulation can be applied with the
help of group and networks, however, these approaches need relationship data. In case the Yelp
dataset employed in this project lacks credible reviewer and time columns, text classification will
be a justifiable primary focus, whereas burst detection would be an addition to the project in the
future. This stance brings about methodological honesty and avoids overclaiming of the project.

2.12 Practical and Ethical Concerns

The literature also reveals that ethical implications of fake review detection exist. A classifier may
be useful to track activities on the platform, but it may also inaccurately classify real customer
experiences as fake ones. Chen et al. (2025) address detection performance in terms of accuracy
and F1-score, which are not the most effective metrics to assess reputational damage due to false
positives. This restriction is significant as moderation of reviews has an impact on both users and
businesses, as well as public confidence. Kumar (2020) demonstrates that fake reviews may
misguide customers, but still, the solution should not be to consider all the suspicious reviews as
fraudulent. Thus, the current paper presents the model as a decision-support artefact, as opposed
to an enforcement tool.
The other useful consideration is interpretability. The traditional models are simpler to describe
since the TF-IDF characteristics can display powerful words, whilst the RNN and Bi-LSTM
models work with learned sequence representations. Mohawesh et al. (2021) endorse contextual
modelling, but more detailed models could decrease the transparency of non-technical
stakeholders. This brings about trade-offs between performance and explainability. In the case of
an information systems project, the most complex model is not necessarily the best solution; a
model that yields useful, testable and understandable results. The dissertation hence presents
various metrics, visualises the confusion matrices and a prediction function in that way that the
artefact may be inspected in practice.

2.13 Chapter Summary

This chapter summarised the current studies on fake review detection and demonstrated that the
research has shifted the focus of linguistic characteristics and classical classifiers to deep-learning,
attention mechanisms and behavioural-network analysis. It is evident that literature concurs that it
is important to have fake review detection, although it also exposes divergence on the most
effective source of evidence. Abri et al. (2020) advocate linguistic indicators whereas Mohawesh
et al. (2021) call on the need to more thoroughly represent the context. Kumar (2020) and He et
al. (2024) demonstrate that behaviour and network structure are value-added, and the techniques
need data, which might not be available in the chosen dataset. The technical background of Bi-
LSTM-based fake review detection by Chen et al. (2025) is the best, however, the PosAtt-BiLSTM
model they offered is more complex than the one of this project. The chapter thus warrants a
specific implementation with the help of Yelp review text, baseline machine learning, RNN and
Bi-LSTM. The following chapter discusses the methodology, preparation of the dataset, model
development and evaluation procedure to put into practice the proposed fake review monitoring
system.

Chapter 3: Research Methodology

3.1 Introduction

In this chapter, the author describes the research process to design and test the fake review system.
The research takes a quantitative experimental design since the project involves the need to make
measurable comparisons of text classification models. Its execution is against a single Yelp-based
dataset to ensure uniformity between data preparation, model training and evaluation. The
approach is structured on the premise of supervised classification as text of the review is
considered as the input and either the fake or authentic tag is considered the target of the variable.
This method is appropriate as fake review detection can be approached as a text recognition and
classification task that relies on the design of features, choice of labelled data and recognition
model choice (Mewada and Dewang, 2022).

3.2 Research Design

The type of research design is an applied experimental design and artefact development. The
experimental section makes a comparison between the traditional machine learning baselines and
the RNN and Bi-LSTM models. The artefact component delivers a working notebook which loads
data, cleans review text, creates features, trains models and evaluates performance and generates
predictions. It is a suitable design since text classification studies have preprocessing requirements
that are repeatable, train-test splits and objective measures are mandated, as opposed to judgement.
It is also important that a baseline model is used since a deep learning model cannot be considered
as more superior unless it is compared with simplier models in identical conditions (Riekert et al.,
2021).

3.4 Tools and Technologies

The project was implemented using Python language which is widely favoured in Applied
Research Projects. which provides extensive support for data cleaning, natural language processing
and machine or deep learning through mature libraries such as pandas, scikit-learn, and
TensorFlow or Keras. It has readable syntax and also the extensive community documentation that
makes it well suited for an MSc-level applied artefact that needs to be reproducible and easy to
review.
The execution was carried out entirely through Google Colab, a cloud-based Jupyter notebook
environment. The reason behind choosing Google Colab is that ,it helps to remove the need for
local environment setup and it also provides free access to GPU acceleration, which is useful for
training the RNN and Bi-LSTM models.
The dataset together with the supplementary files was stored and managed using Google Drive,
since it integrates directly with Colab and this enables the notebook to read and write the data
without it being necessary to carry out local downloads. The dissertation content was first drafted
and structured using Google Docs before it was formatted into the final Word document.

3.4 Sampling and Data Source

The analysis analyses one Yelp-based dataset, which is downloaded on SciDB. The data is chosen
because the Yelp review data is pertinent to the monitoring of online reviews and has been utilized
in studies regarding fake reviews. The first step of implementation checks the dataset structure to
determine the presence of columns like review text, label, rating, date, reviewer and business
identifiers. The core modelling however, only relies on review text and class label. In case there
are other behavioural fields, they will be stored to undergo descriptive analysis, but not considered
as burst variables. This choice is not excessively claiming since acquiring labelled reviews and
feature suitability are the challenges known in false review studies (Mewada and Dewang, 2022).

3.5 Data Cleaning and Preparation

The process of data preparation is initiated by loading the downloaded file, de-compression of
compressed formats and the appropriate review file being selected. It is standardised with column
names being in lower case. Cleaning eliminates duplicate entries, blank review text, invalid and
useless rows and invalid text entries. The text is changed to lowercase, web links and HTML tags
are stripped out, punctuation is de-punctuated and extra space is normalised. Descriptive features
are generated, such as the number of characters, amount of words, the number of sentences and
the mean length of words. All these measures are required due to the possibility of feature
representation to influence the accuracy of text classification, which is influenced by the size of
training sets and preprocessing decisions (Riekert et al., 2021).

3.6 Feature Engineering

The traditional machine learning models utilize features that are based on TF-IDF since it
transforms the cleaned review text into weighted numeric features that can be used in Logistic
Regression, Naive Bayes and SVM. TF-IDF is suitable as it decreases the effect of very frequent
terms and elevates the meanings of those terms that are unique to documents. In the case of neural
models, the text is tokenised and changed into word sequences and padded to a set length which
is fixed. This is necessary since the RNN and Bi-LSTM models handle sequences and not sparse
term-weight vectors. The literature on text classification demonstrates that both the traditional
method of vectorisation and the deep representations are valid, which depends on the complexity
of tasks and resources at hand (Minaee et al., 2021).

3.7 Model Development

The modelling phase has three levels. The baseline classifiers, which are Logistic Regression,
Naive Bayes and Linear SVM are first trained. Second, the basic sequential model is a
conventional RNN that is trained. Third, a Bi-LSTM model is implemented as the primary model
as it is able to comprehend data in both forward and backward directions of a review. The RNN
and Bi-LSTM models have embedding layers, dropout and early stopping to minimize overfitting.
This is staged design that facilitates fair comparison since all the models are using the same cleaned
data, target variable and train-test split. The beauty of deep learning models is that they require
controlled training and validation since the complexity of the model can grow to cause overfitting
(Alzubaidi et al., 2021).
3.8 Evaluation Strategy
The data is split into a training and testing set with an artificial random state in a way that the
outcomes can be reproduced. Accuracy, precision, recall, F1-score and ROC-AUC are used to
evaluate the model performance. Accuracy indicates an overall correctness whereas precision and
recall indicate classification balance. F1-score is also provided since it is a measure of precision
and recall that is merged into a single balance measure. Confusion matrices depict the correct and
incorrect classifications, whereas ROC curves are utilized when there are the binary probability
scores. Various metrics are necessary since the surveys on text classification indicate that it is not
possible to interpret model performance based on only one score (Gasparetto et al., 2022).
22
The model performance is measured against the expected benchmark as indicated in the proposal.
The approximate classification accuracy is 80-82%, meaning that the model should be able to
correctly classify fake and genuine reviews in 80-82 out of 100 test cases. The use of accuracy
alone is not considered since it can be misleading in the event of an imbalanced dataset. Thus, the
ability of the model to be reliable is also evaluated using precision, recall, F1-score and ROC-
AUC. The ultimate outcome is to be interpreted on comparing the obtained accuracy to the
anticipated range of 80-82.
3.9 Artefact Output
The artefact is a Google Colab notebook which generates clean data, exploration graphics, trained
models, tables of model comparison and prediction. The notebook stores model-ready data, final
metrics and visualisation files and trained model objects. The prediction function takes as input
some new review text and produces the prediction of the classes with a confidence rating. This
renders the artefact useful as it shows how the system may aid in fake reviews monitoring.
Nevertheless, the model is outlined as something that helps to make decisions, rather than
something that can be used as a moderation authority. These systems are also sensitive since the
created and fake reviews may be hard to distinguish between actual customer text (Salminen et al.,
2022).
3.10 Ethical and methodological limitations
The project does not involve the gathering of primary data through human participants but is based
on secondary data only. The identifiers of users are not used to identify individuals, but are
considered only as field of the dataset. The research acknowledges that the burst detection is not
part of the research unless the data includes good fields of the timestamp and the reviewer activity.
Fake review detection can be assisted with the help of graphs and networks, yet they necessitate
relational data, which text data might not provide (Yu et al., 2022).
3.11 Chapter Summary
The approach to implementation was introduced in this chapter. It reasoned the quantitative
experimental design, single Yelp-based dataset, organized cleaning, TF-IDF baselines, RNN, Bi-
LSTM and multi-metric evaluation. The following chapter provides results of implementation and
talks in detail and comprehensively about model performance in the context of scope.
23
Chapter 4: Results and Discussion
4.1 Introduction
The chapter will provide the results obtained by the fake review monitoring notebook implemented
and analyze them. The objective of the results chapter is to demonstrate how the Yelp-based dataset
was handled, how the models did perform and whether the obtained results were in line with the
proposed target benchmark of 80-82 percent accuracy. It was implemented in a monitored
classification strategy wherein the primary input to the method was review text and the target class
was fake or genuine review status. The reason behind this is that fake review detection can not
only be considered a technical classification problem, but a trust problem in digital commerce.
Karmakar and Hawkins (2024) elaborate that online trade highly depends on user-generated
reviews and thus needs systems of automatic monitoring and detection of reviews.
Data cleaning outputs, exploratory visualisations, baseline machine learning results, RNN results,
Bi-LSTM results, confusion matrix outputs, ROC visualisation, training curves and saved artefact
files were generated by the completed notebook. In the final notebook, the mode of the dataset was
determined to be review text, the file yelpzip.csv was identified, 60,000 rows were used to model
and the target label was found and the Bi-LSTM was determined as the best model in the automated
final summary.
4.2 Dataset Loading and Structure
The YelpZip review dataset file was the dataset used in the final execution. The implementation
was able to identify the most important fields needed in the project. The columns were identified
as text column, label column, rating column, date column and reviewer column respectively
identified as user:id, date, label, rating, and text. But the business column did not appear in the last
file structure.
This finding is significant as the study can be conducted over different scopes based on the fields
of the dataset available. The existence of review text and labels facilitates the supervised
classification of fake reviews, and the lack of a business column constrained the business-level
behavioural analysis. User and date fields were also included in the dataset, however, the modelling
process was still text-based, as the essence of the actual research was to compare the traditional
machine learning, RNN and Bi-LSTM models. The text preparation output stated that the text
24
mode of implementation was run, text was used as the review column and 250,000 rows were
loaded prior to cleaning.
This validates the fact that the dataset was apt to perform the primary classification task.
Nonetheless, it also validates one of the primary limitations: the research is unable to assume the
complete fake review monitoring by bursts or even by business levels due to the absence of
business identifiers in the identified structure. Thus, the ultimate outcomes can only be seen as
review-text classification outcomes, rather than a full-fledged behavioural fraud detection system.
4.3 Data Cleaning Results
Cleaning of the data enhanced dataset before modelling. The notebook eliminated blank, duplicate
and unusable text documents. The cleaning summary revealed that the dataset contained 250 000
rows, and 249 048 rows after cleaning, which resulted in the elimination of 952 rows. The data
had been cleaned, and at this point, it had 14 columns.
Figure 1: Data cleaning results
The low number of removals indicates that the dataset was mostly usable. The percentage of the
data that was removed was very low, and this implies that most of the review information was
retained during the implementation. Following the preparation of the features, the resulting final
feature dataset had 249,048 rows, 10 columns, a target variable available and the mode of the
dataset was still review text.
This is good in terms of results since cleaning did not cause extreme loss of data which influenced
the performance of the model. Nonetheless, a high model accuracy cannot be ensured by clean
data only. Reviews that are fake are hard to detect as they may be linguistically similar to honest
25
reviews. Recent research on artificial intelligence generated hotel reviews also demonstrates that
counterfeit reviews are becoming more and more hard to regulate since artificial text has many
similarities with the writing of genuine users. A collection of 10,000 real and 10,000 AI-generated
fake hotel reviews by Ignat, Xu and Mihalcea (2024) revealed that the performance of deception
detection can depend on the sentiment, location and language.
4.4 Exploratory Data Analysis Results
The exploratory data analysis was able to give a summary of the review structure, missing values,
class distribution, ratings, counts of words and text features. The notebook produced several visual
outputs and the final conclusion was 19 saved figures.
Figure 2: Target distribution
The analysis of missing values confirmed the validity of the cleaned data set as the essential fields
needed to be classified were present. The distribution of the target classes revealed that there were
both fake and real reviews in the dataset. This was necessary as it is impossible to have binary
classification done without both classes. It was also possible to use a labelled target that enabled
the notebook to continue with supervised learning instead of unsupervised anomaly detection.
26
Figure 3: Rating distribution
Visualisation of the word count and character count revealed that the length of review texts was
different in the dataset. This is important as shorter reviews can have less linguistic evidence
whereas longer reviews can include more beneficial sequential patterns to RNN and Bi-LSTM
models. Nevertheless, the length of a review will not be sufficient to detect fake reviews. It has the
ability to aid feature comprehension, yet the model requires to learn more profound patterns within
text content. This is the reason why both TF-IDF features were used and sequential deep learning
models.
Figure 4: Word count distribution
The rating distribution was also helpful in going to context. Ratings can facilitate descriptive
knowledge of review behaviour, but were not employed as the primary modelling aspect since the
dissertation is about review text classification. This is suitable since the high or low rating does
27
not necessarily indicate whether a review is a fake review. The EDA thus aided the modelling
process but not the model-based classification.
4.5 Model Preparation Results
A subsample of the cleaned data of 60, 000 rows was controlled in the modelling stage. This was
an expedient move since, neural models would take longer to be trained on the entire data and
would consume higher computing power. The dataset was divided into 48,000 records of training
of both features and labels and 12,000 records of testing features and labels.
The TF-IDF representation generated 30,000 text features using the training data and used the
same representation to the testing data. In the case of deep learning, the notebook utilized 5 epochs,
a batch size of 128, an embedding dimension of 128 and a class weight of 1.0 between two classes.
This design provided that each and every model was tested on the same core dataset and test set.
The accuracy factor would not have been sufficient as there are two types of risk in the fake review
detection: the false alarms (false positives) and false negatives. Ferrer (2022) maintains that
different metrics should be used to evaluate classification systems since accuracy, F-score, ROC-
AUC and score-based measures are different metrics used to measure model performance.
4.6 Classic Machine Learning Performance
The conventional machine learning models provided a valuable starting point to the research. The
highest traditional performance was obtained with Logistic Regression (67.53% accuracy, 67.24%
precision, 68.38% recall, 67.81% F1-score and 0.7423 ROC-AUC). Naive Bayes followed closely
with 66.89% accuracy, 66.56% precision, 67.90% recall, 67.22% F1-score, and 0.7317 ROC-AUC.
Linear SVM achieved 64.28% accuracy, 64.11% precision, 64.90% recall, 64.50% F1-score, and
0.6992 ROC-AUC.
28
Figure 5: Traditional Machine Learning Results
These findings indicate that the conventional models have been average. The best balance was
obtained with Logistic Regression since it gave the best accuracy, F1-score and ROC-AUC. Naive
Bayes was a little weaker but still competitive and in this implementation, Linear SVM had the
least performance in traditional performance. This outcome indicates that the TF-IDF properties
were able to retrieve some useful text patterns, but not all to achieve anticipated 80-82 percent
mark.
The balance of behaviour in the classification report of Logistic Regression indicated a balanced
behaviour in both classes. For class 0, precision was 0.68, recall was 0.67, and F1-score was 0.67.
For class 1, precision was 0.67, recall was 0.68, and F1-score was 0.68. Its total accuracy was 0.68
with 12,000 test records.
This balance is beneficial since it indicates that there was no strong preference of one particular
class by Logistic Regression. Nevertheless, the performance level is not high as per the proposal.
Hence, the baseline score shows that the dataset does have learnable text patterns, although these
patterns are too weak to be detected with high accuracy when it comes to fake review detection
with the help of TF-IDF only.
4.7 RNN Results
RNN model did not yield better overall results as compared to the traditional baselines. The RNN
classification report indicated that there was 0.56 accuracy on the test set. In the case of class 0,
29
the model had a precision of 0.59, a recall of 0.37 and F1-score of 0.45. For class 1, it achieved
0.54 precision, 0.75 recall, and 0.63 F1-score.
This finding indicates that the RNN was more effective in recognizing class 1 as compared to class
0, although this was at the expense of lower class 0 recall. Practically, the RNN was more eager to
assign reviews to the class 1, and this elevated recall in this class and lowered the balance between
the two classes. This presents a problem to fake review monitoring since a model that is over-
detective of one class can result in excessive false alerts or fail to detect enough real cases.
Even the RNN output demonstrates that the sequential architecture will not necessarily enhance
the performance. Even a simple RNN might not be able to handle longer-range dependencies in
text, and it might fail to extract the true meaning of review language. In this implementation, the
RNN was not as good as Logistic Regression and Naive Bayes. Thus, the outcome does not justify
the final model of a standard RNN to use on this dataset.
4.8 Bi-LSTM Results
Bi-LSTM model was superior to the RNN model and it was chosen to be the best model in the
automated final summary of the notebook. An embedding layer, spatial dropout, a bidirectional
LSTM layer, dropout, a dense layer, and a final sigmoid output to classify into two categories were
used as the architecture.
Bi-LSTM classification report displayed the accuracy of 0.67. For class 0, precision was 0.70,
recall was 0.59, and F1-score was 0.64. For class 1, precision was 0.64, recall was 0.74, and F1-
score was 0.69. The macro averages were 0.67 precision, 0.67 recall and 0.66 F1-score.
These findings indicate that Bi-LSTM performed significantly better than RNN, particularly on
the overall balance of classes. It continued to exhibit greater recall of class 1 compared to class 0,
but the imbalance was not as great as in the RNN model. This confirms the theoretical hypothesis
that the bidirectional sequence learning model has the ability to learn more useful review context
than a basic RNN. But there was no decisive difference between Bi-LSTM and Logistic Regression
in the visible rounded classification reports. Accuracy of Logistic Regression was approximately
0.68 and Bi-LSTM was approximately 0.67. So, one should take the result with caution.
Bi-LSTM was chosen as the best model in the automated notebook summary, yet the detailed
visible metrics indicate that there is a close difference between Bi-LSTM and Logistic Regression.
30
This implies that the conclusion may not be overly assertive that Bi-LSTM was highly successful
compared to any other model. The better explanation is that Bi-LSTM was the most powerful deep
learning model and the final model chosen in the notebook, and Logistic Regression was a
powerful and competitive baseline.
4.9 Comparison to the expectations
It was proposed that the range of accuracy would be about 80-82. The results which were obtained
were below this standard. The highest apparent scores were approximately 67-68 percent accuracy,
regardless of whether it is Logistic Regression or Bi-LSTM. This implies that the implementation
was done at about 12-15 percentage points lower than the anticipated accuracy range.
This does not imply that this implementation was unsuccessful. Rather, it demonstrates that the
initial projected result was optimistic to the end dataset and model set-up. The findings support
that the model was able to learn significant patterns, albeit at a lower level than anticipated. This
could be because of a number of reasons. First, both fake and genuine reviews can be the same
linguistically. Second, the project was modelled by a review text which was the primary input of
modelling and richer behavioural features were constrained. Third, a realistic configuration of the
deep learning models was five epochs. Fourth, the data might have complex/noisy labels.
Assessment must also be done carefully since false classification of reviews may be influenced by
imbalance of classes and by the choice of metric. According to Holzmann and Klar (2024), some
of the metrics commonly used to measure binary classification are prone to underperform in a
situation of imbalanced classes, thus performance should be measured using multiple indicators
instead of one. Thus, the appropriate result statement would be that the model performed
moderately in terms of classification, but failed to reach the desired 80-82% accuracy rate.
The attained accuracy of about 67-69% is to be considered as a natural consequence of a text-only
fake review classification model, as opposed to a coding error. The initial proposal target of 80-82
was a projected standard prior to the inspection and execution of the models in their entirety. The
post-implementation results indicated that the dataset was conducive to the supervised
classification since the review text and labels were present, but the identified structure failed to
establish a business identifier to analyse the business levels through burst analysis. This limited
the model to linguistic patterns primarily, rating of context, date information and user-level fields.
31
Both fake and authentic reviews can be written in the same vocabulary, emotional interests and
sentence structure, thus the model will be confused with well-written fraudulent reviews and the
real customer experience. This is the reason why the outcome was moderate despite the correct
coding and a big dataset. The performance of Logistic Regression was good as TF-IDF reflected
the direct word-level patterns whereas Bi-LSTM did better than RNN since it had reviewed
sequences in both directions. Both approaches however lacked sufficient behavioural evidence to
isolate all the cases of deception. Soldner et al. (2022) also caution that counterfeit review
classifiers may pick up dataset-specific indicators rather than actual indicators of deception. Thus,
the 67-69 % outcome ought to be addressed as a sincere empirical observation that refers to the
existing limit of a review-text-only system.
4.10 Discussion
RQ1 enquired how a Yelp based review dataset can be cleaned and be ready to fake review
classification. The outcomes demonstrate that the data set was loaded successfully, cleaned and
converted to review-text format. The last feature dataset had 249,048 rows, the target variable was
present and modelling sample had 60,000 rows.
RQ2 to be answered was the effectiveness of traditional machine learning models in classifying
fake and genuine reviews based on text features. These findings indicate that the traditional models
had a moderate performance with the Logistic Regression being the best in terms of the
performance as the baseline. Its ROC-AUC of 0.7423 suggests that it had better ranking ability
than Naive Bayes and Linear SVM.
RQ3 was whether RNN and Bi-LSTM models can effectively classify fake and genuine review
texts based on sequential review texts. The RNN achieved low results, 0.56 accuracy and Bi-LSTM
achieved high results, 0.67 accuracy. This supports the idea that Bi-LSTM was more appropriate
than RNN to the current dataset but it failed to give a significant boost to the best traditional model.
RQ4 questioned what model gives the best performance in regard to the chosen evaluation
measures. Bi-LSTM is the best model that was found by the automated summary of the notebook.
Nevertheless, the apparent metric analysis reveals that Logistic Regression and Bi-LSTM were
almost similar. Thus, it is the best conclusion that the final model was Bi-LSTM, and the most
competitive final model was Logistic Regression.
32
These results also indicate that accuracy cannot be used to evaluate model selection. The Bi-LSTM
was chosen as the final model, but the competitiveness of Logistic Regression was very strong,
which indicates that the dataset has strong word-level indicators that TF-IDF could extract. The
reduced performance of RNN demonstrates that simple sequential learning could not be effective
enough to represent the review patterns. The moderate 67-69% result is thus attributed to the type
of data, rather than just the modelling method. As counterfeit and genuine reviews are frequently
alike in wording, ratings, and emotional utterances, a text-based model presents limited evidence
in partitioning deception and authentic experience. This justifies the behavioural indicators that
include reviewer frequency, business-level activity and review timing. Thus, the findings respond
to the research questions in a realistic way by demonstrating the fact that the classification of
review texts is possible as well as indicating that more powerful detection should be based on more
abundant features and broader validation in further applied systems development.
4.11 Practical Interpretation
The findings show that fake reviews can be detected with the help of the review text, but the
obtained accuracy indicates that text-only monitoring does not have endless possibilities. A system
having about 67-68 percent accuracy can be used to do early review screening but it cannot be
considered a fully automatic moderation tool. The model should not automatically remove reviews
flagged by the model, but consider them as suspicious cases to be further checked.
This reality warning is significant since counterfeit reviews are getting more advanced. Gambetti
and Han (2023) demonstrate that it is possible to generate fake restaurant reviews by using
machines and detect them with the help of an AI-based approach, which means that the detection
of fake reviews starts to be considered an adversarial problem, instead of a text classification task.
The existing findings thus point towards an interpretation of decision support. The model can assist
in determining possible review text to focus on, but further behavioural characteristics like
reviewer history, product level activity, business-level review patterns and timestamp bursts would
be required to have a more robust monitoring mechanism.
4.12 Chapter Summary
The results of the implementation of the completed notebook were analysed in this chapter. Data
loading was successful, and the data was cleaned and ready to be supervised by classifying fake
33
reviews. The last modelling model took 60,000 reviews; the reviews were divided into 48,000
training records and 12,000 test records. The old models have shown average performance with
the best baseline coming out as Logistic Regression. RNN performed worse than the other models,
whereas Bi-LSTM was better than RNN and was chosen to be the final model in the notebook
summary.
The obtained accuracy, however, was not as high as 8082% proposal benchmark. The most optimal
results were between 67 and 68 percent that is, the desired result should be revisited or clarified as
a positive starting point. The findings nevertheless offer practical evidence since they indicate that
review-text categorisation can aid in the monitoring of fake reviews, however, the high-
performance might demand more effective behavioural features, more tuning and more
sophisticated language representations.
34
Chapter 5: Conclusion and Recommendation
5.1 Introduction
This chapter completes the research on fake review detection with the aid of RNN and Bi-LSTM
model. The purpose of the project was to design, implement and test an artefact of fake review
detection based on Yelp-based dataset. The foregoing chapters defined the research problem,
literature review, description of the methodology and interpretation of the results. The findings are
summarised in this chapter, the aim and objective of the study is assessed, the contribution is
discussed and limitations and recommendations are provided.
5.2 Overall Conclusion
The research came up with a full-fledged fake review monitoring implementation. The last
notebook loaded YelpZip dataset, determined the columns of text and label, cleaned the review
text, generated visualisations, trained baseline machine learning models and tested RNN and Bi-
LSTM models. The last running made sure that the mode of the dataset was review text, file was
yelpzip.csv, target label existed, there were two classes, and 60,000 reviews were utilized to train
the model.
The results show that the project achieved its practical implementation aim. The system could
categorize reviews as fake and genuine, compare a number of models and choose a final model.
Nevertheless, performance was not as high as the proposed 80-82% accuracy. The best observable
performance was in the range of 67-68, i.e., the result was average. This finding demonstrates that
it is feasible to detect fake reviews by text of reviews, though it can be difficult when the language
patterns of fake and real reviews are similar.
5.3 Achievement of Objectives
The initial goal was to survey the recent studies concerning fake reviews detection and sequence-
based text classification. This was done by the literature review, which demonstrated that the
traditional machine learning, RNN, Bi-LSTM and behavioural approaches are all relevant. The
second goal was to prepare and clean Yelp dataset to classify fake reviews. This was attained
because the data set of 250 000 rows was narrowed down to 249 048 cleaned rows after deleting
unusable data. The third goal was to use baseline machine learning models. This has been done
with the help of Logistic Regression, Naive Bayes and Linear SVM.
35
The fourth aim was to come up with RNN and Bi-LSTM. The two models were put in place and
tested. The fifth goal was to compare performance of the models based on their accuracy, precision,
recall, F1-score and ROC-AUC. This was done, with the traditional models being evaluated based
on these metrics, and deep learning models evaluated based on classification reports and test
performance. The sixth target was to speak about the worth and shortcomings of the counterfeit
review surveillance strategy. This has been done by clarifying that the model can be used to
facilitate the review screening and cannot be enforced to be used as an enforcement tool.
5.4 Key Findings
The initial important result is that the dataset was appropriate to be supervised to review-text
classification since it had review text and label fields. This was what enabled binary classification
models to be trained. The second observation is that data cleaning did not eliminate numerous
records and this implies that the dataset could be used. Findings three; Logistic Regression had the
best traditional baseline with accuracy and F1-score of about 67.53 and 0.678 respectively. This
indicates that TF-IDF features depicted patterns of texts.
The fourth observation is that the basic RNN model was not very successful with an approximate
accuracy of 56 percent. This shows that a simple sequential model could not be used with this
dataset. The fifth observation is that Bi-LSTM was superior to RNN and was chosen as the ultimate
model in the notebook summary. However, the difference between Bi-LSTM and Logistic
Regression was not very significant, and the outcome should be viewed with caution. The sixth
conclusion is that the predicted 80-82% accuracy level was not achieved. Thus, the expectation
about the proposal has to be a sort of starting point and not a final decision.
5.5 Contribution
The key input of the research is the creation of a structured working fake review monitoring
artefact. The notebook offers a complete pipeline which consists of loading, cleaning, EDA,
feature preparation, model training, evaluation, best model selection, and prediction. This is
helpful since it shows how it is possible to apply fake review classification in an unambiguous and
replicable manner. The research is also valuable as it demonstrates that Bi-LSTM can outperform
a simple RNN, but conventional models might also be competitive in the case when the data is in
textual form.
36
5.6 Limitations
The principal weakness is that the ultimate accuracy was not as high as anticipated. This can be
attributed to the similarity of texts between fake and real reviews, noise in the dataset, limited
tuning, and lack of more behavioural features. Another limitation is that the business column was
not identified, and this curtailed the business-level or burst-pattern analysis. Even though there
were user and date fields, the implementation was primarily centered around review text. The deep
learning models were trained with five epochs, and it was feasible with Colab, but it might not be
the most optimal performance. The research has an authentic artefact, yet not an optimised system.
Another weakness is that the model was tested using a single YelpZip data, which limits the
extrapolation of the results. A model that is trained and evaluated on a specific dataset can
potentially be taught platform-related writing behaviour, instead of general cues of fake review
behaviour. The modelling sample, to be more precise, was scaled to 60,000 records to run the
project on Google Colab, therefore, the findings are an academic controlled implementation, but
not a deployment on a large-scale platform. Moreover, the deep learning models were trained over
five epochs which was suitable to time and resource constraints, but might not reflect the maximum
possible performance. Weaknesses in business identifiers also inhibited more group-level analysis,
although the literature demonstrates that behavioural and network characteristics may enhance
fake review detection in the presence of the appropriate relational data (He et al., 2024). Such
constraints justify the fact that the ultimate accuracy was low than the anticipated level.
5.7 Recommendations
The future work will enhance the dataset and modelling strategy. First, other behavioural features
must be added where possible, including: reviewer history, review frequency, business identifiers,
rating deviation, and posting patterns. Second, additional hyperparameter optimization of Bi-
LSTM must be done, such as sequence lengths, embedding dimensions, dropout rates, epochs, and
batch sizes. Third, models based on transformers could be experimented with, as they might be
able to capture more contextual meaning. Fourth, cross-validation or an external dataset should be
used to test the generalisability of the model. Lastly, the system is to be treated as a decision-
support tool, whereby the suspicious reviews are marked to be reviewed by people instead of being
deleted.
37
5.8 Future work
The system could be enhanced by enriching the future work with more powerful behavioural and
contextual features. If the business identifiers can be found in a different version of a dataset, the
model is supposed to compute the review bursts, the repeat activity of a reviewer, the abnormal
rating variation and the abnormal posting frequency of each business. Such signals can identify
manipulation that is coordinated and cannot be observed using text alone. Transformer-based
models (BERT or RoBERTa) should also be evaluated in the next stage of development since they
have the capacity to capture more meaning of the sentence and contextual relations. Nevertheless,
the models must be compared to the Logistic Regression and Bi-LSTM through the same train-
test split so that no unfair performance claims can be made. Lastly, the system is to be tested on a
separate data set and then declared platform-ready. This would demonstrate how the model can
generalise outside of YelpZip and whether the predicted 80-82% benchmark can be possible with
the addition of more powerful features, longer training, and more powerful tuning. This aids in
testing in the future.
5.9 Final Summary
Finally, the study applied a fake review monitoring model through a Yelp review text model and a
classification model. The last model comparison demonstrated moderate performance, where the
final model was Bi-LSTM, and the baseline was strong due to Logistic Regression. It failed to
meet the anticipated 80-82% accuracy, but the project still came up with a full artefact. The
research shows that monitoring of the fake reviews is possible, although it needs more reliable
data, which involves deeper tuning and more extensive validation.
38
Appendices
Appendix A: Supervisor Meeting / Monitoring Progress Report
This appendix records the four supervision meetings held with the supervisor, Swati Dongre,
throughout the course of the Applied Research Project, including the topics discussed and the
agreed actions arising from each session, as documented on the Applied Research Project
Meeting/Monitoring Progress Report.
Topics Discussed Discussed the research proposal
and received feedback on several
required changes; agreed to
identify and secure approval for a
suitable dataset.
Meeting Supervisor Outcome / Agreed Action
1 Swati
Dongre
Proposal revised per feedback;
student to source a candidate
dataset for supervisor approval.
2 Swati
Dongre
Confirmed the chosen YelpZip
dataset; reviewed whether burst-
pattern (behavioural) detection
was feasible given the available
fields.
Dataset approved; agreed that
burst-pattern detection would be
treated as future work, since the
dataset structure could not
support it.
3 Swati
Dongre
Reviewed implementation gaps;
went through the coding and
results in detail and identified
areas needing revision in the
artefact.
Agreed changes to the code
implemented to address the
identified gaps in the artefact.
4 Swati
Dongre
Discussed dissertation formatting
requirements; finalised the
artefact and reviewed the
supporting documentation.
Formatting corrections agreed;
artefact and documentation
finalised for submission.
39
Appendix B: Sample Dataset Records
A small sample of records from the YelpZip review dataset is presented below to illustrate the raw
data structure prior to cleaning, including the review text, associated rating, and the fake/genuine
label field discussed in Sections 3.3 and 3.4.
https://www.scidb.cn/en/detail?dataSetId=43d7f503936f45acafc2864f6573237b#p2
Appendix C: Code Repository / Notebook Access
The complete implementation, including data cleaning, feature engineering, model training and
evaluation code, is provided as a Google Colab notebook. The notebook link is included here rather
than reproducing the full code and console output, in line with academic guidance to avoid pasting
raw code listings into the appendices.
https://colab.research.google.com/drive/1qA0FtaTkCIVfl-p_9_rVDyS63HpYodM4?usp=sharing
https://github.com/Fluvia20079668/Monitoring-and-Controlling-Fake-Reviews-Using-RNN-
and-Bi-LSTM/blob/main/ba26_0633_.py
Appendix D: Extended Results Tables
This appendix presents additional model evaluation detail supplementary to Chapter 4, including
full classification reports and confusion matrix breakdowns for the Logistic Regression, Naive
Bayes, Linear SVM, RNN and Bi-LSTM models, summarised in tabular form.
RNN Result
Bi-LSTM Result
40
Appendix E: Glossary of Terms and Fiqures of each results
RNN: Recurrent Neural Network. Bi-LSTM: Bidirectional Long Short-Term Memory. TF-IDF:
Term Frequency-Inverse Document Frequency. ROC-AUC: Receiver Operating Characteristic -
Area Under Curve. EDA: Exploratory Data Analysis.
Missing value Visualisation
Find the top 12 colunms with most missing data and create ,labels, saves and display as a bar chat
Fequent word visualisation
examine how many words are present in each review also check the length of the review text.
41
word count by class visualisation
the boxplot to compare the distribution of review word counts between the different target classes.
It allows to see the central tendency and spread of review lengths and compare the classes.
Monthy Review volume Visualisation
to analyse how the volume of reviews changes over time. The reviews are grouped by month and
the number of reviews in each month is plotted as a line graph.
42
Corrrelation Visualisation
the relationships between numerical features such as review length, word count, sentence count,
average word length, rating, and the target variable. The correlation heatmap provides a visual way
to identify strong, weak, positive, or negative relationships between these variables.
Train Test Split
Train test split is the process where we split the data into two as 80% for training and 20 % for
testing.
43
References
Abri, F., Gutierrez, L.F., Namin, A.S., Jones, K.S. and Sears, D.R.W. (2020) ‘Fake reviews
detection through analysis of linguistic features’, arXiv preprint, arXiv:2010.04260. [Online].
Available at: https://arxiv.org/abs/2010.04260 (Accessed: 10 August 2026).
Alam, M., Cevallos, B., Flores, O., Lunetto, R., Yayoshi, K. and Woo, J. (2021) ‘Yelp dataset
analysis using scalable big data’, arXiv preprint, arXiv:2104.08396. [Online]. Available at:
https://arxiv.org/abs/2104.08396 (Accessed: 10 August 2026).
Alzubaidi, L., Zhang, J., Humaidi, A.J., Al-Dujaili, A., Duan, Y., Al-Shamma, O., Santamaría, J.,
Fadhel, M.A., Al-Amidie, M. and Farhan, L. (2021) ‘Review of deep learning: concepts, CNN
architectures, challenges, applications, future directions’, Journal of Big Data, 8(53). doi:
10.1186/s40537-021-00444-8.
Cao, C., Li, S., Yu, S. and Chen, Z. (2021) ‘Fake reviewer group detection in online review
systems’, arXiv preprint, arXiv:2112.06403. [Online]. Available at:
https://arxiv.org/abs/2112.06403 (Accessed: 10 August 2026).
Chen, J., Zhang, T., Yan, Z., Zheng, Z., Zhang, W. and Zhang, J. (2025) ‘Attention-based BiLSTM
with positional embeddings for fake review detection’, Journal of Big Data, 12(83). doi:
10.1186/s40537-025-01130-9.
Ferrer, L. (2022) ‘Analysis and comparison of classification metrics’, arXiv preprint,
arXiv:2209.05355. [Online]. Available at: https://arxiv.org/abs/2209.05355 (Accessed: 11 August
2026).
Gambetti, A. and Han, Q. (2023) ‘Combat AI with AI: counteract machine-generated fake
restaurant reviews on social media’, arXiv preprint, arXiv:2302.07731. [Online]. Available at:
https://arxiv.org/abs/2302.07731 (Accessed: 11 August 2026).
Gasparetto, A., Zangari, A., Marcuzzo, M. and Albarelli, A. (2022) ‘A survey on text classification:
practical perspectives on the Italian language’, PLOS ONE, 17(7), e0270904. doi:
10.1371/journal.pone.0270904.
44
Gutierrez-Espinoza, L., Abri, F., Namin, A.S., Jones, K.S. and Sears, D.R.W. (2020) ‘Fake reviews
detection through ensemble learning’, arXiv preprint, arXiv:2006.07912. [Online]. Available at:
https://arxiv.org/abs/2006.07912 (Accessed: 10 August 2026).
He, L., Wang, X., Chen, H. and Xu, G. (2022) ‘Online spam review detection: a survey of
literature’, Human-Centric Intelligent Systems, 2, pp. 14–30. doi: 10.1007/s44230-022-00001-3.
He, S., Hollenbeck, B., Overgoor, G., Proserpio, D. and Tosyali, A. (2024) ‘Detecting fake review
buyers using network structure: direct evidence from Amazon’, arXiv preprint, arXiv:2410.17507.
[Online]. Available at: https://arxiv.org/abs/2410.17507 (Accessed: 10 August 2026).
Holzmann, H. and Klar, B. (2024) ‘Robust performance metrics for imbalanced classification
problems’, arXiv preprint, arXiv:2404.07661. [Online]. Available at:
https://arxiv.org/abs/2404.07661 (Accessed: 11 August 2026).
Ignat, O., Xu, X. and Mihalcea, R. (2024) ‘MAiDE-up: multilingual deception detection of GPT-
generated hotel reviews’, arXiv preprint, arXiv:2404.12938. [Online]. Available at:
https://arxiv.org/abs/2404.12938 (Accessed: 11 August 2026).
Karmakar, P. and Hawkins, J. (2024) ‘Enhanced review detection and recognition: a platform-
agnostic approach with application to online commerce’, arXiv preprint, arXiv:2405.06704.
[Online]. Available at: https://arxiv.org/abs/2405.06704 (Accessed: 11 August 2026).
Kennedy, S., Walsh, N., Sloka, K., Foster, J. and McCarren, A. (2020) ‘Fact or factitious?
Contextualized opinion spam detection’, arXiv preprint, arXiv:2010.15296. [Online]. Available at:
https://arxiv.org/abs/2010.15296 (Accessed: 10 August 2026).
Kumar, J. (2020) ‘Fake review detection using behavioral and contextual features’, arXiv preprint,
arXiv:2003.00807. [Online]. Available at: https://arxiv.org/abs/2003.00807 (Accessed: 10 August
2026).
Mewada, A. and Dewang, R.K. (2022) ‘Research on false review detection methods: a state-of-
the-art review’, Journal of King Saud University - Computer and Information Sciences, 34, pp.
7530–7546. doi: 10.1016/j.jksuci.2021.07.021.
Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M. and Gao, J. (2021) ‘Deep
learning-based text classification: a comprehensive review’, ACM Computing Surveys, 54(3),
45
Article 62. doi: 10.1145/3439726. [Online]. Available at: https://arxiv.org/abs/2004.03705
(Accessed: 10 August 2026).
Mohawesh, R., Xu, S., Springer, M., Al-Hawawreh, M. and Maqsood, S. (2021) ‘Fake or genuine?
Contextualised text representation for fake review detection’, arXiv preprint, arXiv:2112.14343.
[Online]. Available at: https://arxiv.org/abs/2112.14343 (Accessed: 10 August 2026).
Riekert, M., Riekert, M. and Klein, A. (2021) ‘Simple baseline machine learning text classifiers
for small datasets’, SN Computer Science, 2(178). doi: 10.1007/s42979-021-00480-4.
Salminen, J., Kandpal, C., Kamel, A.M., Jung, S. and Jansen, B.J. (2022) ‘Creating and detecting
fake reviews of online products’, Journal of Retailing and Consumer Services, 64, 102771. doi:
10.1016/j.jretconser.2021.102771.
Salunkhe, A. (2021) ‘Attention-based bidirectional LSTM for deceptive opinion spam
classification’, arXiv preprint, arXiv:2112.14789. [Online]. Available at:
https://arxiv.org/abs/2112.14789 (Accessed: 10 August 2026).
Shahariar, G.M., Biswas, S., Omar, F., Shah, F.M. and Hassan, S.B. (2022) ‘Spam review detection
using deep learning’, arXiv preprint, arXiv:2211.01675. [Online]. Available at:
https://arxiv.org/abs/2211.01675 (Accessed: 10 August 2026).
Soldner, F., Kleinberg, B. and Johnson, S. (2022) ‘Confounds and overestimations in fake review
detection: experimentally controlling for product-ownership and data-origin’, PLOS ONE, 17(12),
e0277869. doi: 10.1371/journal.pone.0277869.
Yu, S., Ren, J., Li, S., Naseriparsa, M. and Xia, F. (2022) ‘Graph learning for fake review
detection’, Frontiers in Artificial Intelligence, 5, 922589. doi: 10.3389/frai.2022.922589.
46
