import cv2
import numpy as np
import os


# 读取图片
def read_image(image_path):
    # 以彩色图像模式读取图片
    image = cv2.imread(image_path)
    return image


# 计算非黑色像素的比例
def calculate_non_black_ratio(image):
    # 将图像转换为灰度图，方便处理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将所有白色像素（值为255）转换为完全黑色（值为0）
    gray_image[gray_image == 255] = 0

    # 设置阈值，假设阈值为127，低于127的像素值设置为0，高于127的像素值设置为255
    threshold_value = 65
    max_value = 255

    # 使用cv2.threshold进行二值化操作
    _, gray_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)

    # # 创建一个可调节大小的窗口
    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # # 调整窗口大小为宽600，高600
    # cv2.resizeWindow('Image', 600, 600)
    # # 显示图片
    # cv2.imshow('Image', gray_image)
    # # 等待按键按下，按下任意键继续
    # cv2.waitKey(0)
    # # 关闭所有窗口
    # cv2.destroyAllWindows()

    # 找到所有像素值不为0的像素点，0表示完全黑色
    non_black_pixels = np.sum(gray_image != 0)

    # 计算图像的总像素数
    total_pixels = image.shape[0] * image.shape[1]

    # 计算非黑色像素的比例
    ratio = non_black_pixels / total_pixels
    return ratio


import pandas as pd

def process_folder(folder_path):
    results = []  # 用于存储文件路径和非黑色像素比例的列表

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpg"):  # 只处理.png文件
                image_path = os.path.join(root, filename)
                image = read_image(image_path)

                # 检查图像是否成功读取
                if image is None:
                    print(f"无法读取文件: {image_path}")
                    continue

                # 计算非黑色像素比例
                non_black_ratio = calculate_non_black_ratio(image)

                # 将结果添加到列表中
                results.append({"文件路径": image_path, "非黑色像素比例": non_black_ratio})

    # 将结果保存到CSV文件
    df = pd.DataFrame(results)
    df.to_csv(folder_path + "detection_results.csv", index=False)
    print("检测结果已保存到 detection_results.csv")




def main():
    folder_path = '20241024'  # 替换为你的文件夹路径
    process_folder(folder_path)


# 运行程序
if __name__ == '__main__':
    main()
