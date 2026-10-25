bias, swing = self.voltages()
p_cw = dbm_to_w(p.laser_power_dbm) * 10 ** (-p.insertion_loss_db / 10)
e_in = cw_laser(self.rng, drive.shape, p_cw, p.linewidth_hz, p.rin_db_hz, fs)
push = p.drive == "push_pull"
field = e_in * mzm_field(bias + swing * drive, p.v_pi, p.extinction_ratio_db, push)
