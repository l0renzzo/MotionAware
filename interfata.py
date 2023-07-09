import tkinter as tk
from tkinter import ttk
import threading
import app_monitoring
from tkinter.filedialog import askopenfilename


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MotionAware')
        self.geometry('600x600')
        self.resizable(0, 0)
        self.style=ttk.Style(self)
        self.configure(bg='LightBlue1')
        self.style.theme_use("clam")
        self.create_header_frame()
        self.thread_squat_monitoring = None
        self.thread_desk_monitoring = None
        self.thread_camera_calibration = None
        self.app_monitoring = None
        self.file_with_camera_params = None

    def create_header_frame(self):
        self.s = ttk.Style(self)
        self.s.configure("My.TFrame", background ="LightBlue1")
        self.s.configure('TButton', background='LightBlue1')
        self.s.configure('TButton', foreground='black')
        self.s.configure("TMenubutton", background="LightBlue1")
        self.header = ttk.Notebook(self)
        
        self.tab1 = ttk.Frame(self.header, style = "My.TFrame")
        self.header.add(self.tab1, text='Calibrate Camera')
        
        self.tab2 = ttk.Frame(self.header, style = "My.TFrame")
        self.header.add(self.tab2, text='Desk monitoring')

        self.tab3 = ttk.Frame(self.header, style = "My.TFrame")
        self.header.add(self.tab3, text='Squat monitoring')

        self.header.pack(expand=1, fill="both")

        self.shoulder_ankle = tk.StringVar()
        self.var1 = tk.IntVar()
        self.no_photos_var = tk.StringVar()
        self.no_rows_var = tk.StringVar()
        self.no_cols_var = tk.StringVar()

        ########################        Calibrate camera       #####################################

        self.label1 = ttk.Label(self.tab1, text="No of photos: ",background='LightBlue1', width= 22, font=("arial",10,"bold"))
        self.label1.place(x=260, y=140)
        self.entry1=ttk.Entry(self.tab1, font=(30), width = 10, textvariable=self.no_photos_var)
        self.entry1.place(x=260, y=160)

        self.button1=tk.Button(self.tab1, text='Take pictures', width = 30, command=self.take_photos)
        self.button1.place(x=200, y=200)

        self.label2 = ttk.Label(self.tab1, text="No of rows: ",background='LightBlue1', width= 22, font=("arial",10,"bold"))
        self.label2.place(x=260, y=260)
        self.entry2=ttk.Entry(self.tab1, font=(30), width = 10, textvariable=self.no_rows_var)
        self.entry2.place(x=260, y=280)

        self.label3 = ttk.Label(self.tab1, text="No of cols: ",background='LightBlue1', width= 22, font=("arial",10,"bold"))
        self.label3.place(x=260, y=310)
        self.entry3=ttk.Entry(self.tab1, font=(30), width = 10, textvariable=self.no_cols_var)
        self.entry3.place(x=260, y=330)

        self.button2=tk.Button(self.tab1, text='Re-calibrate', width = 30, command=self.calibrate_camera)
        self.button2.place(x=200, y=370)

        self.button12=tk.Button(self.tab1, text='Choose calibration parameters file', width = 30, command=self.choose_calibration_file)
        self.button12.place(x=200, y=450)

        ########################        Desk monitoring       #####################################

        self.button10=tk.Button(self.tab2, text='Left side', width = 20, command=lambda: self.change_side('left'))
        self.button10.place(x=100, y=120)

        self.button11=tk.Button(self.tab2, text='Right side', width = 20, command=lambda: self.change_side('right'))
        self.button11.place(x=300, y=120)

        self.button3=tk.Button(self.tab2, text='Start', width = 20, command=self.desk_monitoring_start)
        self.button3.place(x=200, y=200)

        self.button4=tk.Button(self.tab2, text='Re-calibrate', width = 20, command=self.desk_monitoring_recalibrate)
        self.button4.place(x=200, y=230)

        self.button5=tk.Button(self.tab2, text='Stop', width = 20, command=self.monitoring_stop)
        self.button5.place(x=200, y=260)

        ########################        Squat       #####################################

        self.button6=tk.Button(self.tab3, text = 'Start', width = 20, command=self.squat_monitoring_start)
        self.button6.place(x=200, y=200)

        self.button7=tk.Button(self.tab3, text='Stop', width = 20, command=self.monitoring_stop)
        self.button7.place(x=200, y=260)

        self.button8=tk.Button(self.tab3, text='Left side', width = 20, command=lambda: self.change_side('left'))
        self.button8.place(x=100, y=120)

        self.button9=tk.Button(self.tab3, text='Right side', width = 20, command=lambda: self.change_side('right'))
        self.button9.place(x=300, y=120)

    #       functions for camera calibration        #
    def take_photos(self):
        no_photos = self.no_photos_var.get()
        if not no_photos.isnumeric():
            return
        no_photos_int = int(no_photos)
        if no_photos_int < 1:
            return
        self.app_camera_calibration = app_monitoring.CameraCalibration()
        self.thread_camera_calibration = threading.Thread(target=self.app_camera_calibration.take_photos, args=(no_photos_int,))
        self.thread_camera_calibration.start()

    def calibrate_camera(self):
        no_photos = self.no_photos_var.get()
        no_rows = self.no_rows_var.get()
        no_cols = self.no_cols_var.get()
        if not no_photos.isnumeric() or not no_rows.isnumeric() or not no_cols.isnumeric():
            return
        
        no_photos_int = int(no_photos)
        no_rows_int = int(no_rows)
        no_cols_int = int(no_cols)
        if no_photos_int < 1 or no_rows_int < 1 or no_cols_int < 1:
            return

        self.app_camera_calibration = app_monitoring.CameraCalibration()
        self.thread_camera_calibration = threading.Thread(target=self.app_camera_calibration.calibrate_camera, args=(no_photos_int, no_rows_int, no_cols_int))
        self.thread_camera_calibration.start()
        print('calibrating')

    def choose_calibration_file(self):
        self.file_with_camera_params = askopenfilename(filetypes=(("NPZ files", "*.npz"),))
        self.file_with_camera_params = self.file_with_camera_params.split('/')[-1]
        print(self.file_with_camera_params)
    #       end functions for camera calibration        #

    #       functions for DESK monitoring        # 
    def desk_monitoring_start(self):
        if self.file_with_camera_params is None:
            return
        self.monitoring_stop()
        self.app_monitoring = app_monitoring.DeskMonitoring(self.file_with_camera_params)
        self.thread_desk_monitoring = threading.Thread(target=self.app_monitoring.start_application)
        self.thread_desk_monitoring.start()
        print('Desk monitoring...')
        
    
    def desk_monitoring_recalibrate(self):
        print('Recalibrating...')
        if self.app_monitoring is not None:
            self.app_monitoring.recalibrate = True
    #       end functions for desk monitoring        #

    #       functions for SQUAT monitoring        # 
    def squat_monitoring_start(self):
        if self.file_with_camera_params is None:
            return
        self.monitoring_stop()
        self.app_monitoring = app_monitoring.SquatMonitoring(self.file_with_camera_params)
        self.thread_squat_monitoring = threading.Thread(target=self.app_monitoring.start_application)
        self.thread_squat_monitoring.start()
        print('Squat monitoring...')
        
    def monitoring_stop(self):
        if self.app_monitoring is not None:
            self.app_monitoring.run = False
            self.app_monitoring = None
    #       end functions for SQUAT monitoring        #

    def change_side(self, side):
        self.squat_monitoring_side = side
        if self.app_monitoring is not None:
            self.app_monitoring.selected_side = side


if __name__ == "__main__":
    app = Application()
    app.mainloop()
