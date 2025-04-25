class Utils:
    @staticmethod
    def to_np_img(img):
        img = img.permute(1, 2, 0)
        img = img.cpu().numpy()

        return img
    