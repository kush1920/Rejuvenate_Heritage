import cv2
import numpy as np


class SequentialTransform:
    def __init__(self, geometric_transforms,out_size):
        self.geometric_transforms = geometric_transforms
        self.out_size = out_size

    def _get_transformation_matrix(self, img_size):
        w, h = img_size
        T = np.identity(3)
        for transform in self.geometric_transforms:
            T = np.matmul(transform.get_transformation_matrix((w, h)), T)
        return T

    def apply_transform(self, masked , mask , ori,
                        interpolation=cv2.INTER_AREA,
                        border_mode=cv2.BORDER_CONSTANT,
                        border_value=(127, 127, 127)):

        h, w = masked.shape[:2]
        T = self._get_transformation_matrix(img_size=(w, h))
        out1 = cv2.warpPerspective(masked.copy(), T, self.out_size, None, interpolation, border_mode, border_value)
        out2 = cv2.warpPerspective(mask.copy(), T, self.out_size, None, interpolation, border_mode, border_value)
        out3 = cv2.warpPerspective(ori.copy(), T, self.out_size, None, interpolation, border_mode, border_value)

        return out1 , out2 , out3


