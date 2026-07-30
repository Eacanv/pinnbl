import numpy as np
import tensorflow as tf

class BrooksCorey:
    """Brooks-Corey relative permeability model"""
    def __init__(self, swc=0.2, snwr=0.2, nw=2.0, nnw=2.0, 
                 krwmax=0.3, krnwmax=0.8, muw=1.0e-3, munw=5.0e-3):
        self.swc = swc
        self.snwr = snwr
        self.nw = nw
        self.nnw = nnw
        self.krwmax = krwmax
        self.krnwmax = krnwmax
        self.muw = muw
        self.munw = munw
        
        self.compute_shock_properties()
    
    def normalized_saturation(self, S):
        """Compute normalized saturation"""
        S_star = (S - self.swc) / (1.0 - self.swc - self.snwr)
        S_star = tf.clip_by_value(S_star, 0.0, 1.0)
        return S_star
    
    def kr_water(self, S):
        """Water relative permeability"""
        S_star = self.normalized_saturation(S)
        return self.krwmax * tf.pow(S_star, self.nw)
    
    def kr_nonwater(self, S):
        """Non-water relative permeability"""
        S_star = self.normalized_saturation(S)
        return self.krnwmax * tf.pow(1.0 - S_star, self.nnw)
    
    def fractional_flow(self, S):
        """Fractional flow function fw(S)"""
        krw = self.kr_water(S)
        krnw = self.kr_nonwater(S)
        eps = 1e-10
        M = (krnw / (krw + eps)) * (self.muw / self.munw)
        fw = 1.0 / (1.0 + M)
        return fw
    
    def compute_shock_properties(self):
        """Compute shock saturation using Welge tangent construction"""
        S_vals = np.linspace(self.swc + 0.001, 1.0 - self.snwr, 1000)
        S_tensor = tf.constant(S_vals.reshape(-1, 1), dtype=tf.float32)
        
        fw_vals = self.fractional_flow(S_tensor).numpy().flatten()
        
        # Get fw at connate saturation
        S_swc_tensor = tf.constant([[self.swc]], dtype=tf.float32)
        fw_swc = self.fractional_flow(S_swc_tensor).numpy()[0, 0]
        
        # Find shock saturation
        slopes = (fw_vals - fw_swc) / (S_vals - self.swc)
        idx_shock = np.argmax(slopes)
        
        self.sw_tangent = S_vals[idx_shock]
        self.fw_tangent = fw_vals[idx_shock]
        self.fw_swc = fw_swc
        self.alpha_tangent = (self.fw_tangent - fw_swc) / (self.sw_tangent - self.swc)
        
        print(f"\nShock properties (Welge tangent):")
        print(f"  fw(swc): {fw_swc:.4f}")
        print(f"  Shock saturation (sw_tangent): {self.sw_tangent:.4f}")
        print(f"  fw at shock: {self.fw_tangent:.4f}")
        print(f"  Shock slope (alpha): {self.alpha_tangent:.4f}\n")
        
class analytic_solution():
    def __init__(self, Swc, Sor, krw0, kro0, nw, no, mu_w, mu_o, 
                 Nx, Nt, A, qt, phi, fw_tangent, sw_tangent, L, T):
        self.Swc = Swc
        self.Sor = Sor
        self.krw0 = krw0
        self.kro0 = kro0
        self.nw = nw
        self.no = no
        self.mu_w = mu_w
        self.mu_o = mu_o
        self.Nx = Nx
        self.Nt = Nt
        self.A = A
        self.qt = qt
        self.phi = phi
        self.L = L
        self.T = T
        
        self.fw_tangent = fw_tangent
        self.sw_tangent = sw_tangent
        self.dx = L / (Nx-1)
        self.dt = T / (Nt-1)
        
        self.v_shock = qt / A / phi * (fw_tangent)/(sw_tangent-Swc)
        self.ttf = np.arange(self.Nt) * self.dt
        self.xf = self.v_shock * self.ttf
        
        self.x_test = np.arange(Nx) * self.dx
    
    # Saturación efectiva
    def Swe(self, Sw):
        return np.clip((Sw - self.Swc) / (1 - self.Swc - self.Sor), 0, 1)
    
    # Permeabilidades relativas
    def krw(self, Sw):
        return self.krw0 * self.Swe(Sw) ** self.nw
    
    def kro(self, Sw):
        return self.kro0 * (1 - self.Swe(Sw)) ** self.no
    
    def fw(self, Sw):
        return (self.krw(Sw) / self.mu_w) / (self.krw(Sw) / self.mu_w + self.kro(Sw) / self.mu_o)
    
    def dfw_dSw(self, Sw, eps=1e-6):
        return (self.fw(Sw + eps) - self.fw(Sw - eps)) / (2 * eps)
    
    def create_solution(self):
        saturation_profiles = []
        
        for i in range(self.Nt):
            # Check if shock front is within domain
            if self.xf[i] <= self.L:
                # Shock hasn't reached the end yet
                xs = len(self.x_test[self.x_test < self.xf[i]])
                
                if xs > 0:
                    # Create saturation values from 1-Sor down to sw_tangent
                    y = np.linspace(1-self.Sor, self.sw_tangent, xs)
                    
                    # Calculate positions for each saturation
                    s_x = self.dfw_dSw(y) * self.qt / self.A / self.phi * self.ttf[i]
                    
                    # Create interpolation points
                    xvals = np.linspace(0, self.xf[i], xs)
                    
                    # Interpolate saturation at grid points
                    yinterp = np.interp(xvals, s_x, y)
                    
                    # Pad with Swc for the rest of the domain
                    pad_length = self.Nx - len(yinterp)
                    profile = np.pad(yinterp, (0, pad_length), 'constant', constant_values=self.Swc)
                else:
                    # Shock hasn't started yet
                    profile = np.full(self.Nx, self.Swc)
            else:
                # Shock has passed through the domain
                # Calculate saturations at all grid points
                y = np.linspace(1-self.Sor, self.sw_tangent, self.Nx)
                s_x = self.dfw_dSw(y) * self.qt / self.A / self.phi * self.ttf[i]
                
                # Interpolate at actual grid locations
                yinterp = np.interp(self.x_test, s_x, y)
                
                # Set values beyond shock to sw_tangent
                yinterp[self.x_test > self.xf[i]] = self.sw_tangent
                
                profile = yinterp
            
            saturation_profiles.append(profile)
        
        return saturation_profiles