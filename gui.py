from PIL import ImageTk, ImageOps
import PIL.Image
import io
from tkinter import *
import tkinter as tk
import numpy as np
from keras.models import load_model

windo = Tk()
windo.configure(background='white')
windo.title("Digit Recognition")
model = load_model('./model/mnistLearner.h5')

windo.geometry('1120x820')
windo.resizable(0, 0)


def destroy_widget(widget):
    widget.destroy()


def pred_digit():
    global no, no1
    ps = canvas.postscript(colormode='color')
    # use PIL to convert to PNG
    im1 = PIL.Image.open(io.BytesIO(ps.encode('utf-8')))
    img = im1.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    print(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img/255.0
    # predicting the class
    res = model.predict([img])[0]
    pred = np.argmax(res)
    acc = max(res)
    no = tk.Label(windo, text='Predicted Digit is: '+str(pred), width=34, height=1,
                  fg="white", bg="midnightblue",
                  font=('times', 16, ' bold '))
    no.place(x=460, y=380)

    no1 = tk.Label(windo, text='Prediction Accuracy is: '+str(acc), width=34, height=1,
                   fg="white", bg="red",
                   font=('times', 16, ' bold '))
    no1.place(x=460, y=415)


def draw_digit(event):
    #    canvas.configure(background="black")
    x = event.x
    y = event.y
    r = 10
    canvas.create_oval(x-r, y-r, x + r, y + r, fill='black')
    panel5.configure(state=NORMAL)


def clear_digit():
    panel5.configure(state=DISABLED)
    canvas.delete("all")
    try:
        no.destroy()
        no1.destroy()
    except:
        pass


panel5 = Button(windo, text='Predict Digit', state=DISABLED, command=pred_digit,
                width=15, borderwidth=0, bg='midnightblue', fg='white', font=('times', 18, 'bold'))
panel5.place(x=60, y=305)

panel6 = Button(windo, text='Clear Digit', width=15, borderwidth=0,
                command=clear_digit, bg='red', fg='white', font=('times', 18, 'bold'))
panel6.place(x=60, y=355)

canvas = tk.Canvas(windo, width=380, height=280, highlightthickness=1,
                   highlightbackground="midnightblue", cursor="pencil")
canvas.grid(row=0, column=0, pady=2, sticky=W,)
canvas.place(x=460, y=90)
canvas.bind("<B1-Motion>", draw_digit)

lab = tk.Label(windo, text="Draw Digit", width=18, height=1, fg="white", bg="midnightblue",
               font=('times', 16, ' bold '))
lab.place(x=554, y=60)

windo.mainloop()
