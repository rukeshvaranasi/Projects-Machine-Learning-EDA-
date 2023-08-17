import streamlit as st
from PIL import Image
import pickle
import numpy as np

# Load the trained models
loaded_knn_model = pickle.load(open(r'G:\python applications\cv_proj\resources\knn_model.sav', 'rb'))
loaded_lr_model = pickle.load(open(r'G:\python applications\cv_proj\resources\lr_model.sav', 'rb'))
loaded_dt_model = pickle.load(open(r'G:\python applications\cv_proj\resources\dt_classifier.sav', 'rb'))
loaded_rf_model = pickle.load(open(r'G:\python applications\cv_proj\resources\rf_classifier.sav', 'rb'))

def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert the image to grayscale
    image = image.convert("L")

    # Convert the image to a numpy array and flatten it
    image_array = np.array(image).flatten()

    return image_array

def make_prediction(model, image_array):
    # Make the prediction using the model
    prediction = model.predict(image_array.reshape(1, -1))
    return prediction[0]


# Define the main Streamlit app
def main():
    # Load custom CSS file for styling
    custom_css = open('custom.css', 'r').read()
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)
    # Page title 
    st.title("Recognizing Handwritten Alphabets")

    # Sidebar with app description and instructions
    st.sidebar.header("About this App")
    st.sidebar.write("This app uses different machine learning models to predict handwritten alphabets.")
    st.sidebar.write("Upload an image, choose a model, and see the predicted alphabet.")
    st.sidebar.write("The app supports PNG, JPG, and JPEG formats.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        
        #preprocess the image
        test_image=preprocess_image(image)



        # Model selection
        model_options = {
            "KNN Classifier Model": loaded_knn_model,
            "Logistic Regression Model": loaded_lr_model,
            "Decision Tree Classifier Model": loaded_dt_model,
            "Random Forest Classifier Model": loaded_rf_model
        }

        # Define the accuracies for each model 
        accuracies = {
        "KNN Classifier Model": 95.869535,
        "Logistic Regression Model": 87.645119,
        "Decision Tree Classifier Model": 94.829938,
        "Random Forest Classifier Model": 98.640362
        }


        selected_model = st.selectbox("Select a model for predicting the alphabet:", list(model_options.keys()))

        st.subheader("Model Description")
        if selected_model == "KNN Classifier Model":
            st.write("K-Nearest Neighbors (KNN) is a simple and widely used classification algorithm.")
            st.write(f"Accuracy of {selected_model} in predicting the handwritten alphabet from the images : <span class='accuracy-highlight'> {accuracies[selected_model]:.2f}%</span>", unsafe_allow_html=True)
        elif selected_model == "Logistic Regression Model":
            st.write("Logistic Regression is a linear classifier used for binary and multiclass classification.")
            st.write(f"Accuracy of {selected_model} in predicting the handwritten alphabet from the images : <span class='accuracy-highlight'> {accuracies[selected_model]:.2f}%</span>", unsafe_allow_html=True)
        elif selected_model == "Decision Tree Classifier Model":
            st.write("Decision Tree is a non-linear classifier that splits the data based on features to make decisions.")
            st.write(f"Accuracy of {selected_model} in predicting the handwritten alphabet from the images : <span class='accuracy-highlight'> {accuracies[selected_model]:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.write("Random Forest is an ensemble learning method that combines multiple decision trees for improved accuracy.")
            st.write(f"Accuracy of {selected_model} in predicting the handwritten alphabet from the images : <span class='accuracy-highlight'> {accuracies[selected_model]:.2f}%</span>", unsafe_allow_html=True)
        # Make the prediction using the selected model
        prediction = make_prediction(model_options[selected_model], test_image)

        # Display prediction and probability
        st.subheader(f"Predicted Alphabet using {selected_model}")
        st.write(f"The alphabet predicted by the model : <span class='prediction-highlight'>{prediction}</span>", unsafe_allow_html=True)
            

if __name__ == "__main__":
    main()