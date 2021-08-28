# SCAAI 2nd Round project
**TOPIC:**  
Developing a image classifiaction model for classifying the disaster into different classes
of disaster, like land disaster, water disaster etc.

**Project structure**

```
-- <SCAAI_Drive_Anindyadeep_Sannigrahi_PS3>
        |_______ code
                    |___ progress_test.py
                    |___ results.csv
                    |___ run.py
        |_______ Comprehensive Disaster Dataset(CDD)
                    |__ Damaged_Infrastructure
                    |__ Fire_Disaster
                    |__ Human_Damage
                    |__ Land_Disaster
                    |__ Non_Damage
                    |__ Water_Disaster
        |_______ IPYNB
                    |__ FINAL_APPROACH_FILE_MAKING.ipynb
                    |__ RESNET152_final224by224.ipynb
                    |__ VGG19_bn.ipynb
                    |__ VGG19.ipynb
        |_______ models
                    |__ RESNET152.pth
                    |__ VGG19_bn.pth
                    |__ VGG.pth
        |_______ rough tests
                    |__ model testing-resenet152.ipynb
                    |__ model testing-vgg19_bn.ipynb
                    |__ model testing-vgg19.ipynb
        |_______ test_folder
        |_______ final.zip
        |_______ Readme.md
        |_______ Requirements.txt
```

<br>

## **How to run the files and test the models with unkown paths**
**STEPS FOR CONFIGURING THE TEST FOLDER**

1.  **[WAY 1]:** You can either go to the test_folder and copy paste the images those are being collected, <br>
    and paste those images there and our `run.py` file will process those images automatically and will provide
    the outputs


2.  **[WAY 2]:** You can also remove the pre-existing `test_folder` and paste the required folder of user that will<br>
    be used for the testing purposes, just rename that folder to `test_folder`. But the folder must be placed in the main<br>
    folder `<SCAAI_Drive_Anindyadeep_Sannigrahi_PS3>` removing the old `test_folder` folder. Otherwise it wll not work.

<br>
<br>

## **Running the code commands**

I have made my own test images which are in the `test_folder` that I have used to anaylyse my model. You can just see the <br>
accuracy of the model on the basis of that test data provided by me by going to the `code` subfolder and there you can run this <br>
following code as shwown below:

```bash
python3 -c 'import os; import run; print(run.get_accuracy_test_results())' 
```

For running the model for an unknwon dataset provided by the user, can be run through this command below. <br>
**NOTE:** The folder `test_folder` must be at the right place.

```bash
python3 -c 'import os; import run; path = str(os.getcwd())[:-4]+"test_folder"; print(run.get_test_results(path))'
```