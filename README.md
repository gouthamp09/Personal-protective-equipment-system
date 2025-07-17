# Personal-protective-equipment-system

problem statement

Many construction workers fail to wear essential safety gear like helmets, safety vests, and masks. Manual monitoring is inefficient, costly, and error-prone, making it difficult to ensure consistent safety compliance on busy sites. Existing systems lack real-time reporting, automated violation alerts, and identity recognition features, reducing their effectiveness in preventing accidents. This gap highlights the need for an intelligent, automated system that can monitor, detect, and report safety violations instantly while identifying the responsible individuals.

motivation

Construction sites are prone to accidents due to the lack of proper PPE (Personal Protective Equipment) usage, leading to serious injuries and safety hazards. Our goal is to enhance worker safety by automatically detecting PPE compliance in real-time and providing alerts for immediate corrective action. To achieve this, we use YOLOv8, a cutting-edge object detection model, to identify essential safety equipment such as helmets, vests, and masks from both live video feeds and uploaded images. Additionally, the system integrates InsightFace for face recognition, allowing us to associate detected violations with specific individuals, thereby increasing personal accountability. Whenever a violation is detected, the system sends automated email alerts to site supervisors with details like the worker’s name, timestamp, and a captured image of the incident. 

abstract

Our project proposes a real-time safety detection system for construction sites using YOLOv8 for detecting PPE like helmets, vests, and masks from both live video streams and uploaded images. It integrates InsightFace for face recognition and email alert notifications to supervisors when violations occur, thereby improving site safety and accountability. The system ensures continuous, automated monitoring without the need for manual supervision, reducing human error and operational costs. It also maintains detailed records of violations, including worker identity and timestamps, providing valuable data for safety audits and decision-making. Designed to be scalable and adaptable, the system can be extended to include additional safety rules and detection features in the future

