import streamlit as st
import os
from PIL import Image
import cv2
import argparse
import os
import glob
import random
import time
import numpy as np
import darknet

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov3-tiny-object_final-crop.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov3-tiny-object-crop.cfg", #TODO
    # parser.add_argument("--config_file", default="./yolov3-tiny-object-crop.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data", # TODO
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    return batch_array

def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
    else:
        image = image_or_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    batch_array = prepare_batch(images, network)
    batch_array = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    darknet_images = darknet.IMAGE(image_width, image_height, 3, batch_array)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))





# Initialize session state for page navigation

if 'page' not in st.session_state:
    st.session_state.page = 1

# Set page layout
st.set_page_config(layout="wide")

# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Initialize session state variables
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False


uploaded_file_path = "uploads/image.jpg"
captured_file_path = "uploads/image.jpg"

# Function to delete existing files
def clear_upload_folder():
    for filename in os.listdir("uploads"):
        file_path = os.path.join("uploads", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Page 1: Media Upload

def page1():
    custom_css = """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        margin: 0;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(75.7deg, rgb(34, 126, 34) 3.8%, rgb(99, 162, 17) 87.1%);
        height: 100%;
    }
    </style>
    """

    # Inject the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    image_path = "reversed.jpg"
    image = Image.open(image_path)

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url('data:Background_image/jpg;base64');
        background-size: cover;
        background-repeat: no-repeat;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 style='text-align: center; font-size: 80px;'>Crop Disease Detection</h1>", unsafe_allow_html=True)

    dotted_line = """<div style="border-top: 2px dotted #00FFFF; margin-top: 20px; margin-bottom: 20px;"></div>"""
    st.markdown(dotted_line, unsafe_allow_html=True)
    st.subheader("Your Intelligent Crop Disease Detection Assistant")

    # Define columns
    col1, col2 = st.columns([1, 2])

    # Display image in the first column
    with col1:
        st.image(image, use_column_width=True)

    # Display how it works in the second column
    with col2:

        st.markdown("### How It Works")
        st.markdown("<p style='font-size:22px;'>• Capture or Upload: Take a photo or upload an image of your crop.</h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:22px;'>• AI Analysis: Our advanced AI analyzes the image for signs of disease.</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:22px;'>• Get Results: Receive a detailed report with diagnosis and treatment recommendations.</h1>",
            unsafe_allow_html=True)

        st.markdown("<div style='height: 65px;'></div>", unsafe_allow_html=True)

        if st.button("Start Analysis"):
            st.write("Starting analysis...")
            st.session_state.page = 2
            # st.experimental_rerun() # TODO
            st.rerun()


def page2():
    custom_css = """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        margin: 0;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(75.7deg, rgb(34, 126, 34) 3.8%, rgb(99, 162, 17) 87.1%);
        height: 100%;
    }
    </style>
    """

    # Inject the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 80px;'>Crop Disease Detection</h1>", unsafe_allow_html=True)
    dotted_line = """<div style="border-top: 2px dotted #00FFFF; margin-top: 20px; margin-bottom: 20px;"></div>"""
    st.markdown(dotted_line, unsafe_allow_html=True)
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)


    st.subheader("Media Upload")
    st.caption("Add your photo here, and you can upload up to 1 photo max")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "svg"], key="file_upload")
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            clear_upload_folder()
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded and saved as image.jpg")

        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

        if st.button('Previous Page'):
            st.session_state.page = 1
            # st.experimental_rerun()n # TODO
            st.rerun()


    with col2:
        st.markdown('<div class="camera-container">', unsafe_allow_html=True)
        # dotted_line = """<div style="border-top: 2px dotted #000000; margin-bottom: 0px;"></div>"""
        # st.markdown(dotted_line, unsafe_allow_html=True)
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
        st.markdown("📷")
        if st.button("Take a picture", key="camera_button"):
            st.session_state.show_camera = True
        if st.session_state.show_camera:
            captured_image = st.camera_input("Take a picture", key="camera_input")
            if captured_image is not None:
                clear_upload_folder()
                with open(captured_file_path, "wb") as f:
                    f.write(captured_image.getbuffer())
                st.success("Captured image saved as image.jpg")
                st.session_state.show_camera = False

        st.markdown("<div style='height: 105px;'></div>", unsafe_allow_html=True)
        if st.button("Next"):

            st.session_state.page = 3
            # st.experimental_rerun() # TODO
            st.rerun()
            
def page3():
    custom_css = """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        margin: 0;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(75.7deg, rgb(34, 126, 34) 3.8%, rgb(99, 162, 17) 87.1%);
        height: 100%;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    images = load_images(args.input)

    index = 0

    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = "uploads/image.jpg"
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )

        print(detections)

        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        if not args.dont_show:
            cv2.imwrite('uploads/a.jpg',image)
    
            break


    if os.path.exists(image_name) and detections!=[]:
        disease = detections[0]
        if disease[0]=='Having_Disease':
            image = Image.open(image_name)
            col1, col2 = st.columns([1.5, 3])
            with col1:
                if st.button('Previous Page'):
                    st.session_state.page = 2
                    # st.experimental_rerun() # TODO:
                    st.rerun()

                st.image('uploads/a.jpg', use_column_width=True)
                st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

                if st.button('Solutions'):
                    st.markdown("<p style='font-size:22px;'>•Use disease-free seeds and plants.</p>",
        unsafe_allow_html=True
    )
                    st.markdown("<p style='font-size:22px;'>•Rotate crops to avoid disease buildup in soil.</p>",
        unsafe_allow_html=True
    )
                    st.markdown("<p style='font-size:22px;'>•Practice good sanitation (removing plant debris, disinfecting tools).</p>",
        unsafe_allow_html=True
    )
                    st.markdown("<p style='font-size:22px;'>•Use appropriate fungicides or bactericides where necessary.</p>",
        unsafe_allow_html=True
    )
                    st.markdown("<p style='font-size:22px;'>•Ensure proper plant spacing and air circulation</p>",
        unsafe_allow_html=True
    )
                    st.markdown("<p style='font-size:22px;'>•Avoid overhead watering to reduce leaf wetness.</p>",
        unsafe_allow_html=True
    )
            with col2:
                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:55px; padding-left: 50px;'>• Alert: Disease Detected</p>",
        unsafe_allow_html=True
    )
     
                st.markdown("<p style='font-size:44px; padding-left: 51px;'> Causes:</p>",
        unsafe_allow_html=True
    )
                st.markdown("<p style='font-size:22px; padding-left: 50px;'>•All these diseases thrive in environments with high humidity and moisture, including prolonged leaf wetness or poorly drained soil conditions. Wet conditions favor the growth and spread of fungal and bacterial pathogens.</p>",
        unsafe_allow_html=True
    )
                st.markdown("<p style='font-size:22px; padding-left: 51px;'>•Most of these pathogens thrive in warm temperatures, which create favorable conditions for their growth and spread.</p>",
        unsafe_allow_html=True
    )
                st.markdown("<p style='font-size:22px; padding-left: 51px;'>•Using infected seeds, plants, or plant debris can introduce pathogens into new crops.These pathogens can survive on plant material and contaminate healthy plants.</p>",
        unsafe_allow_html=True
    )
                st.markdown("<p style='font-size:22px; padding-left: 51px;'>•Lack of proper sanitation, such as not removing diseased plant debris, using contaminated tools, or not disinfecting equipment, can lead to the spread of pathogens across plants and fields.</p>",
        unsafe_allow_html=True
    )
                st.markdown("<p style='font-size:22px; padding-left: 51px;'>•Wind, rain, irrigation splash, and contaminated water are common dispersal mechanisms for both fungal spores and bacteria, facilitating the spread of disease across crops.</p>",
        unsafe_allow_html=True
    )
                st.markdown("<p style='font-size:22px; padding-left: 51px;'>•Continuous cropping of susceptible plants without rotating to non-host crops can lead to a buildup of pathogens in the soil, increasing disease pressure over time.</p>",
        unsafe_allow_html=True
    )
      
        else:
            col1, col2 = st.columns([1.5, 3])
            with col1:
                if st.button('Previous Page'):
                    st.session_state.page = 2
                    # st.experimental_rerun() # TODO
                    st.rerun()
                st.image('uploads/a.jpg', use_column_width=True)



            with col2:
                st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
                st.markdown(
          "<p style='font-size:50px; padding-left: 50px;'>•Invalid Input or Disease not detected.</h1>",
            unsafe_allow_html=True)


    elif detections==[]:
        col1, col2 = st.columns([1.5, 3])
        with col1:
            if st.button('Previous Page'):
                st.session_state.page = 2
                # st.experimental_rerun() #TODO
                st.rerun()
                
            st.image('uploads/a.jpg', use_column_width=True)



        with col2:
            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
            st.markdown(
          "<p style='font-size:50px; padding-left: 50px;'>•Invalid Input or Disease not detected.</h1>",
            unsafe_allow_html=True)






if st.session_state.page == 1:
    page1()
elif st.session_state.page == 2:
    page2()
elif st.session_state.page == 3:
    page3()
