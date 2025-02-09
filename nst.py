import re
import torch
from torchvision import transforms
from transformer_net import TransformerNet
import cv2
from PIL import Image

def stylizeImage(content_image, styleno=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = f'data/style{styleno}/nst.model'
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Resize(512)
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model = torch.nn.DataParallel(style_model)
        output = style_model(content_image).cpu()
    return output[0]

def stylizeVideo(input_video_path, output_video_path, styleno=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = f'data/style{styleno}/nst.model'
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    with torch.no_grad():
        state_model = TransformerNet()
        state_dict = torch.load(model)

        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        state_model.load_state_dict(state_dict)
        state_model.to(device)
        style_model = torch.nn.DataParallel(state_model)
    
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                content_image = frame
                content_image = content_transform(content_image)
                content_image = content_image.unsqueeze(0).to(device)
                output = style_model(content_image).cpu()
                output = output.squeeze(0)
                output = output.permute(1,2,0)
                output = output.numpy()
                output = output.astype('uint8')
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                out.write(output)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def stylizeCamera(styleno=2):
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frame = stylizeImage(frame)
        frame = frame.squeeze(0)
        frame = frame.permute(1,2,0)
        frame = frame.numpy()
        frame = frame.astype('uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()