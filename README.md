# project_nsf

Understanding how human vision correlates with Convolutional Neural Networks (CNNs)

---

- nsf\_pipeline - data pipeline that extracts features from Vgg16 for spefic layers, trains an SVM on each layer and computes human correlations between SVM hyperplane distance and human categorization accuracy
- train\_vgg - contains scripts for training Vgg16 from scratch for grayscale and color images

---
Plot of increasing layer complexity, SVM layer accuracy and Spearman's rho correlation with human categorization accuracy 
![caffe_results](https://user-images.githubusercontent.com/16754088/31024905-8c6c6c1c-a50e-11e7-8285-b0ed4136311f.png)

---
Plot of conv5_2 hyperplane distances correlations with human categorization accuracy
![conv5_2_correlations](https://user-images.githubusercontent.com/16754088/31024944-b0480876-a50e-11e7-9058-6da9b6d6037b.png)

