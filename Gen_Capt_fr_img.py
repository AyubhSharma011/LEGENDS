import cv2
from transformers import BlipProcessor,BlipForConditionalGeneration
from PIL import Image
import torch

# OPEN CAMERA AND SAVE PICTURE:
def Capture_frame():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: CAMERA IS NOT FOUND:")
        return
    print("PRESS S TO SAVE IMAGE OR Q TO QUIT:  ")

    while True:
        ret,frame = cap.read()
        if not ret:
            print("FAILED TO GRAB FRAME:    ")
            return
        cv2.imshow("VISION_MATE CAMERA READY:" , frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            Img_Name = "Output.jpg"
            cv2.imwrite(Img_Name, frame)
            print("IMAGE SAVED AS OUTPUT.JPG:")
            return Img_Name

        elif key == ord('q'):
            print("EXIT WITHOUT SAVING:")
            break
    # Final step:
    cap.release()
    cv2.destroyAllWindows()
    return None

# Caption Generation:
def Generate_Caption(Im_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(Im_path).convert('RGB')

    inputs = processor(image,return_tensors = "pt")

    out = model.generate(**inputs)

    caption = processor.decode(out[0] , skip_special_tokens = True)
    print(f"Caption:    {caption}")
    return caption
    
if __name__  ==  "__main__":
    Image_path = Capture_frame()
    if Image_path :
        Generate_Caption(Image_path)