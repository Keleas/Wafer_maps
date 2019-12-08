# Classification of Wafer Maps Defect Based on Deep Learning Methods With Small Amount of Data

IEEE 6th International Conference Engineering & Telecommunication – En&T-2019 (_Forthcoming_)

## Introduction: The Purpose of the Research

Improvement of the quality of pattern recognition method in conditions of a deficient amount of labeled experimental data

## Work Accomplished
- Method of preparing the __composite training dataset__:
  - review of typical __manufacturing causes__ of defect patterns;
  - procedure of __synthesized wafe maps__ creation;
  - adaptive configuration of training dataset.

- New learning DCNN strategy:
  - pretrain stage __on pure synthetic dataset__;
  - main train stage __on composite dataset__.

- Numerical calculations and results:
  - DCNN model training: VGG-19, ResNet-50, ResNet-34 and MobileNetV2;
  - experimental comparison of models accuracy on different conditions;
  - dependence of classification accuracy on amount of experimental data.

## Review of Typical Manufacturing Causes

![review](img/review.png)

[Source of experimental data](mirlab.org/dataSet/public/WM-811K.zip)

## Synthesis of Wafer Maps

![synthesis](img/synthesis.png)

## Experimental Comparison

![dependence](img/dependence.png)

## Accuracy Specification of the Top DCNN Model

![matrix](img/matrix.png)

## Conclusion

- Proposal of the method of preparing the __composite training dataset__

- Development of the __new learning DCNN model strategy__ which improve the final result of accuracy __by 1% up to 4%__

- __Experimental accuracy comparison__ of VGG-19, ResNet-50, ResNet-34 and MobileNetV2 DCNN models for the different ratio of experimental labeled data to synthesized data

- Achievement of __87.8%__ final classification accuracy with Rₗₛ = 0.05 on the public dataset WM-811K by ResNet-50

- Formative evaluation of needed amount of __experimental data__ to obtain __required accuracy__  
