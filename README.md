# Insulator-defect-detection

## Summary
Aiming at the problems of low speed of existing insulator defect detection, low detection accuracy and difficult to be deployed to embedded terminals, a lightweight insulator defect target detection algorithm LiteYOLO-ID based on the improvement of YOLOv5 is proposed. on the IDID-Plus dataset compared with the original YOLOv5s, not only does it reduce the number of model parameters by 47.13%, but also increase the average accuracy ( mAP0.5) is also increased by 1%. After TensorRT optimization, the inference speed of LiteYOLO-ID algorithm on Jetson TX2 NX reaches 20.2FPS, which is 15.56% higher than that of the original YOLOv5s, and it can meet the requirement of real-time insulator defects detection.


## G-C2f Schematic Diagram

## LiteYOLO-ID Schematic Diagram

## Dataset
The full data set will be published later

## Experimental flow chart
![Experimental procedure：](chart_experiment.png)

## Detection result
![Comparison chart of test results：](Insulator_defect_detection_results_chart.png)
