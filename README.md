# LiteYOLO-ID: A Lightweight Object Detection Network for Insulator Defect Detection

Paper submission number TIM-23-08169 was submitted to ieee transactions on instrumentation and measurement.

**Dear reviewers: The source code and pre-trained model weights will be available upon the acceptance of the paper.   Feel free to raise your questions or difficulties in the implementation.**


## Summary
Addressing the issues of low detection speed, suboptimal accuracy, and challenges in deploying existing insulator defect detection methods on embedded terminals, we propose a lightweight insulator defect detection algorithm, LiteYOLO-ID, based on the improvement of YOLOv5. When compared to the original YOLOv5s on the IDID-Plus dataset, our model exhibits a reduction of 47.13% in model parameters and an improvement of 1% in average precision (mAP0.5). Following TensorRT optimization, the LiteYOLO-ID algorithm achieves an inference speed of 20.2 FPS on the Jetson TX2 NX, representing a 15.56% enhancement over the original YOLOv5s. This performance meets the real-time detection requirements for insulator defects.


## G-C2f Schematic Diagram

## LiteYOLO-ID Schematic Diagram

## Dataset
The full data set will be published later

## Experimental flow chart
![Experimental procedure：](chart_experiment.png)

## Detection result
![Comparison chart of test results：](Insulator_defect_detection_results_chart.png)


## Author's Contact
Email：ly13063414159@163.com
