import numpy as np
import MNIST_NN as MN
import BackPropV2 as BP
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tkinter import *
from tkinter import ttk

#import MNIST_NN as NN

canvasDimension = 280 
canvasPixelValues = np.zeros((canvasDimension, canvasDimension))
smallCanvasVal = np.zeros((28, 28))
finalValNumb = 0

def imageScaling():
    conversionRatio = canvasDimension / 28
    for r in range(28):
        for c in range(28):
            xIndex, yIndex = int(conversionRatio * r), int(conversionRatio * c)
            smallCanvasVal[r][c] = canvasPixelValues[xIndex][yIndex]
    
    #filterGaus = np.zeros((3,3))
    #filterGaus = gaussian_filter(filterGaus, sigma=1)

    for r in range(1, 27):
        for c in range(1, 27):
            matrix3x3 = smallCanvasVal[int(r - 1) : int(r + 2), int(c - 1) : int(c + 2)]
            filterGaus = gaussian_filter(matrix3x3, sigma=0.3)
             
            #totalVal = np.sum(filterGaus) #np.sum(matrix3x3)
            #smallCanvasVal[r][c] = int(totalVal)
            smallCanvasVal[int(r - 1) : int(r + 2), int(c - 1) : int(c + 2)] = filterGaus

def displayImage():
    imageScaling()
    fig, Myaxis = plt.subplots(nrows= 1, ncols= 2)
    Myaxis[0].imshow(canvasPixelValues)
    Myaxis[1].imshow(smallCanvasVal)

    plt.show()

def updatePos(event):
        global xVal, yVal
        xVal, yVal = (event.x), (event.y)

def clearImage():
    global canvasPixelValues, smallCanvasVal
    canvas.delete("all")
    canvasPixelValues = np.zeros((canvasDimension, canvasDimension))
    smallCanvasVal = np.zeros((28, 28))

def drawCircle(event):
    drawSize = 20
    xVal, yVal = (event.x + drawSize), (event.y + drawSize)
    canvas.create_rectangle(event.x, event.y, xVal ,yVal ,fill = "black", width=10)
    #updatePos(event)
    canvasPixelValues[event.y : yVal, event.x : xVal] = 255

def makeNNresult():
    global finalValNumb
    imageScaling()
    nodes3 = BP.makeNodes(smallCanvasVal.flatten(), MN.wab)[3]
    finalValNumb = np.argmax(nodes3)
    update2.config(text=(f"OutputValue: {finalValNumb}\n(Keep in mind it may be incorrect\nhopefully not often xd)"))
    print(finalValNumb)

def makeNN():
    MN.neuralNetFunc(int(epoch.get()), float(learningRate.get()), int(bachSize.get()),
                          int(trainingDataSize.get()), int(testingSize.get()))

def drawingInput():
    global canvas, xVal, yVal, trainingDataSize, testingSize, bachSize, learningRate, epoch, finalValNumb
    global update1, update2

    root = Tk()
    root.geometry("800x500")
    root.title("Neural Network Demo :)")

    inputDataFrame = ttk.Frame(root)
    inputDataFrame.grid(row=0, column=0, sticky=(N, S, E, W))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    trainingDataSize = StringVar(value= "5000")
    testingSize = StringVar(value= "5000")
    bachSize = StringVar(value= "20")
    learningRate = StringVar(value= "1.5")
    epoch = StringVar(value= "4")
    trainingEntry = Entry(inputDataFrame, width=5, textvariable=trainingDataSize).grid(row=1,column=1)
    testingEntry = Entry(inputDataFrame, width=5, textvariable=testingSize).grid(row=2,column=1)
    bachSizeEntry = Entry(inputDataFrame, width=5, textvariable=bachSize).grid(row=3,column=1)
    learningRateEntry = Entry(inputDataFrame, width=5, textvariable=learningRate).grid(row=4,column=1)
    epochEntry = Entry(inputDataFrame, width=5, textvariable=epoch).grid(row=5,column=1)
    settingValue = [trainingDataSize, testingSize, bachSize, learningRate, epoch]

    Label(inputDataFrame, text="Fill out the following parameters: ").grid(row=0,column=0)
    Label(inputDataFrame, text="Number of Training Images: ").grid(row=1,column=0)
    Label(inputDataFrame, text="Number of Testing Images: ").grid(row=2,column=0)
    Label(inputDataFrame, text="Batch size: \n(Multiple of #Training Images)").grid(row=3,column=0)
    Label(inputDataFrame, text="Learning Rate: ").grid(row=4,column=0)
    Label(inputDataFrame, text="Epoch: ").grid(row=5,column=0)
    update1 = Label(inputDataFrame, text="Training the network can take \nsome time, check the consol \nto see progress and result")
    update1.grid(row=6,column=0)

    canvas = Canvas(inputDataFrame, width=canvasDimension, height=canvasDimension, background='deepskyblue')
    canvas.grid(row=0,column=2, rowspan=6)
    root.rowconfigure(0, weight=1)

    Button(inputDataFrame, text="Display Input Number", command=displayImage).grid(row=6, column=2)
    Button(inputDataFrame, text="Classify Number", command=makeNNresult).grid(row=8, column=2)
    Button(inputDataFrame, text="Reset Canvas", command= clearImage).grid(row=7, column=2)
    Button(inputDataFrame, text="Submit Settings", command=makeNN).grid(row=6, column=1)
    update2 = Label(inputDataFrame, text=(f"OutputValue: {finalValNumb}\n(Keep in mind it may be incorrect\nhopefully not often xd)"))
    update2.grid(row=9,column=2)

    canvas.bind("<B1-Motion>", drawCircle)
    root.mainloop()

drawingInput()

