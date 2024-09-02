from __future__ import print_function
from numpy import array
from tkinter import *
from fit_line import *
from fit_curve import *

convas_width = 800
convas_height = 800

def transform_coords(x, y, convas_width, convas_height):
    new_x = convas_width // 2 + x
    new_y = convas_height // 2 - y
    return new_x, new_y

# center of bounding box
def cntr(x1, y1, x2, y2):
    return x1+(x2-x1)/2, y1+(y2-y1)/2

# tkinter Canvas plus some addons
class MyCanvas(Canvas):
    def create_polyline(self, points, **kwargs):
        for p1, p2 in zip(points, points[1:]):
            self.create_line(p1, p2, kwargs)

    def create_fit_line(self, k_, b_, tag):
        width = self.winfo_width()
        height = self.winfo_height()
        
         # Calculate where the line intersects the canvas boundary
        if k_ != 0:
            x1 = 0
            y1 = k_ * x1 + b_
            x2 = width
            y2 = k_ * x2 + b_
        else:
           # If the slope k is 0, the line is horizontal
            x1 = 0
            y1 = b_
            x2 = width
            y2 = b_
        # Crop points to stay within canvas range
        points = [(x1, y1), (x2, y2)]
        clipped_points = []
        for x, y in points:
            if 0 <= y <= height:
                clipped_points.append((x, y))
            elif y < 0:
                clipped_points.append((x, 0))
            elif y > height:
                clipped_points.append((x, height))
        
        if len(clipped_points) == 2:
            self.delete(tag)
            self.create_line(clipped_points[0], clipped_points[1], fill='black', width=3, tag=tag)

    def create_fit_curve(self, points_, a_, b_, c_, tag):
        width = self.winfo_width()
        height = self.winfo_height()
        # Calculate where the line intersects the canvas boundary
        if a_ != 0:
            x1 = 0
            y1 = a_ * x1**2 + b_ * x1 + c_
            x2 = width
            y2 = a_ * x2**2 + b_ * x2 + c_
        # Crop points to stay within canvas range
        points = [(x1, y1), (x2, y2)]
        clipped_points = []
        for x, y in points:
            if 0 <= y <= height:
                clipped_points.append((x, y))
            elif y < 0:
                clipped_points.append((x, 0))
            elif y > height:
                clipped_points.append((x, height))

        x_fit = np.linspace(x1, x2, 100)
        for x_tmp in range(x1, x2):
            y_tmp = a_ * x_tmp**2 + b_ * x_tmp + c_
            self.create_oval(x_tmp - 1, y_tmp - 1, x_tmp + 1, y_tmp + 1, outline='green', width=1)
            x_tmp = x_tmp + 1

    def create_point(self, x, y, r, **kwargs):
        return self.create_oval(x-r, y-r, x+r, y+r, kwargs)

    def pos(self, idOrTag):
        return cntr(*self.coords(idOrTag))

    def itemsAtPos(self, x, y, tag):
        return [item for item in self.find_overlapping(x, y, x, y) if tag in self.gettags(item)]


class MainObject:
    def run(self):
        root = Tk()

        self.canvas = MyCanvas(root, bg='white', width=convas_width, height=convas_height)
        self.canvas.pack(side=LEFT)
        # Draw the coordinate axis
        self.canvas.create_line(5, -convas_height, 5, convas_height, width=3, fill='blue')
        self.canvas.create_text(15, convas_height - 20, text='Y', anchor='w')
        self.canvas.create_line(-convas_width, 5, convas_width, 5, width=3, fill='red', tags='y')
        self.canvas.create_text(convas_width - 20, 15, text='X', anchor='w')
        self.canvas.create_oval(-5, -5, 5, 5, outline='green', width=2)

        input_Button = Button(root, text ="Insert Points & Plot Curve", command = self.curve_fit)
        input_Button.pack()

        self.points = []
        self.draggingPoint = None

        self.canvas.bind('<ButtonPress-1>', self.onButton1Press)
        self.canvas.bind('<ButtonPress-2>', self.onButton2Press)
        self.canvas.bind('<B1-Motion>', self.onMouseMove)
        self.canvas.bind('<ButtonRelease-1>', self.onButton1Release)

        root.mainloop()

    def onButton1Press(self, event):
        items = self.canvas.itemsAtPos(event.x, event.y, 'point')
        if items:
            self.draggingPoint = items[0]
        else:
            self.points.append(self.canvas.create_point(event.x, event.y, 4, fill='red', tag='point'))
            self.redraw()
            print(event.x, event.y)

    def onButton2Press(self, event):
        self.canvas.delete(self.points.pop())
        self.redraw()

    def onMouseMove(self, event):
        if self.draggingPoint:
            self.canvas.coords(self.draggingPoint, event.x-4, event.y-4, event.x+4, event.y+4)
            self.redraw()

    def onButton1Release(self, event):
        self.draggingPoint = None

    def onSpinBoxValueChange(self):
        self.redraw()

    def curve_fit(self):
        if len(self.points) < 2:
            return
        self.canvas.delete('bezier')
        points = array([self.canvas.pos(p) for p in self.points])
        tmp_a, tmp_b, tmp_c = fit_curve(points)
        figure = self.canvas.create_fit_curve(points, tmp_a, tmp_b, tmp_c, tag='best_fit_curve')

    def redraw(self):
        if len(self.points) < 2:
            return

if __name__ == '__main__':
    o = MainObject()
    o.run()


