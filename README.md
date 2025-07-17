# Personal-protective-equipment-system

# problem statement

Many construction workers fail to wear essential safety gear like helmets, safety vests, and masks. Manual monitoring is inefficient, costly, and error-prone, making it difficult to ensure consistent safety compliance on busy sites. Existing systems lack real-time reporting, automated violation alerts, and identity recognition features, reducing their effectiveness in preventing accidents. This gap highlights the need for an intelligent, automated system that can monitor, detect, and report safety violations instantly while identifying the responsible individuals.

# motivation

Construction sites are prone to accidents due to the lack of proper PPE (Personal Protective Equipment) usage, leading to serious injuries and safety hazards. Our goal is to enhance worker safety by automatically detecting PPE compliance in real-time and providing alerts for immediate corrective action. To achieve this, we use YOLOv8, a cutting-edge object detection model, to identify essential safety equipment such as helmets, vests, and masks from both live video feeds and uploaded images. Additionally, the system integrates InsightFace for face recognition, allowing us to associate detected violations with specific individuals, thereby increasing personal accountability. Whenever a violation is detected, the system sends automated email alerts to site supervisors with details like the worker’s name, timestamp, and a captured image of the incident. 

# abstract

Our project proposes a real-time safety detection system for construction sites using YOLOv8 for detecting PPE like helmets, vests, and masks from both live video streams and uploaded images. It integrates InsightFace for face recognition and email alert notifications to supervisors when violations occur, thereby improving site safety and accountability. The system ensures continuous, automated monitoring without the need for manual supervision, reducing human error and operational costs. It also maintains detailed records of violations, including worker identity and timestamps, providing valuable data for safety audits and decision-making. Designed to be scalable and adaptable, the system can be extended to include additional safety rules and detection features in the future

Construction Safety Detection - Mail Alert (Yolov8)

This project focuses on enhancing construction site safety through real-time detection of safety gear such as helmets, vests, and masks worn by workers, as well as detecting the presence of a person. The detection is performed using YOLOv8, a state-of-the-art object detection algorithm.
Overview

Construction sites present various safety hazards, and ensuring that workers wear appropriate safety gear is crucial for accident prevention. This project automates the process of safety gear detection using computer vision techniques. By deploying YOLOv8, the system can detect whether a worker is wearing a helmet, a vest, a mask, or all, and identify people in real-time.
Features

    Helmet Detection: Detects whether a worker is wearing a helmet.
    Vest Detection: Detects whether a worker is wearing a safety vest.
    Mask Detection: Detects whether a worker is wearing a mask.
    Person Detection: Detects the presence of a person within the construction site.
    Count Display: Displays real-time counts of detected helmets, vests, masks, and persons on a sideboard overlay.
    Email Alerts: Sends email alerts if a person is detected without a helmet, with a frame of the incident attached.
    Non-Blocking Email Process: Ensures video feed remains smooth while email alerts are sent in the background.
    Mail Sent Notification: A popup is displayed in the top-right corner of the video feed when an email alert is successfully sent.

Requirements

    Python 3.9
    YOLOv8 dependencies (refer to YOLOv8 documentation for installation instructions)
    OpenCV
    Other dependencies as mentioned in the project code


