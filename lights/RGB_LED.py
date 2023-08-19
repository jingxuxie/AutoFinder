#%%
import serial

#%%
class RGBW_LED():
    def __init__(self):
        try:
            self.com = serial.Serial('COM4', 19200, timeout = 2)
            self.color_dict = {'R':' 0', 'G':' 1', 'B':' 2', 'W':' 3'}
        except Exception as e:
            print(e)
            print('RGB light control connection error')

    def RGBW_control(self, RGBW, control):
        '''
        RGBW: str or list
        RGBW: str or list of 'on'/'off'
        '''
        
        assert type(RGBW) == type(control)
        if type(RGBW) == str:
            command = 'relay ' + control + self.color_dict[RGBW] + '\n\r'
            self.com.write(command.encode())
        else:
            assert len(RGBW) == len(control)
            for i in range(len(RGBW)):
                color = RGBW[i]
                state = control[i]
                command = 'relay ' + state + self.color_dict[color] + '\n\r'
                self.com.write(command.encode())
    
    def only_R(self):
        self.RGBW_control(['R', 'G', 'B'], ['on', 'off', 'off'])
    
    def only_G(self):
        self.RGBW_control(['R', 'G', 'B'], ['off', 'on', 'off'])
    
    def only_B(self):
        self.RGBW_control(['R', 'G', 'B'], ['off', 'off', 'on'])

    def only_W(self):
        self.RGBW_control(['R', 'G', 'B', 'W'], ['off', 'off', 'off', 'on'])

    def off_all(self):
        self.RGBW_control(['R', 'G', 'B', 'W'], ['off', 'off', 'off', 'off'])
    
    def on_all(self):
        self.RGBW_control(['R', 'G', 'B', 'W'], ['on', 'on', 'on', 'on'])
    

    def close(self):
        self.com.close()


#%%
if __name__ =='__main__':
    light = RGBW_LED()
# %%
