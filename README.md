# generic_camera_calibration

<br>

- camera intrinsic/extrinsic calibration with opencv

<br>

## **Dataset Structure**

<br>

- **reference**: [link](https://gaussian37.github.io/vision-concept-calibration/#opencv%EC%9D%98-zhangs-method%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%B9%B4%EB%A9%94%EB%9D%BC-%EC%BA%98%EB%A6%AC%EB%B8%8C%EB%A0%88%EC%9D%B4%EC%85%98-%EC%8B%A4%EC%8A%B5-1)
- **dataset**: [link](https://drive.google.com/file/d/1ri4Go75UWQ3JHmRZwWVW35ms3xYosmgf/view?usp=sharing)
- **output example**: [link](https://drive.google.com/file/d/1ri4Go75UWQ3JHmRZwWVW35ms3xYosmgf/view?usp=sharing)

<br>

- `Intrinsic`
    - `ORIGIN`
        - `000.png`
        - `001.png`
        - ...
    - `REPROJECTION` (**OUTPUT**)
        - `000_reprojection_error.png`
        - `001_reprojection_error.png`
        - ...
    - `intrinsic.json` (**OUTPUT**)
- `Extrinsic`
    - `CAMERA_MODEL_NAME_EXTRINSIC.png`
    - `CAMERA_MODEL_NAME`
        - `points.csv`
        - `points.png`
- `CAMERA_MODEL_NAME_calibration.json` (**OUTPUT**)



