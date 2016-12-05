sudo opcontrol --init 
sudo opcontrol --no-vmlinux
sudo opcontrol --reset
sudo opcontrol --start
sh run_ps_mpi_cmd.sh
sudo opcontrol --dump
sudo opcontrol --stop
sudo opcontrol --shutdown
sudo opannotate -s ffm_ps > oprofile_ffm_ps.txt
