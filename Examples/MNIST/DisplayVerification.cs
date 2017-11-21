using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ATTFace
{
    public partial class DisplayVerification : Form
    {
        public DisplayVerification()
        {
            InitializeComponent();
        }


        public void ShowData(List<Bitmap> faces, List<Bitmap> facesTwin, int[] labels, int[] predictions)
        {
            string s;

            //Pair 1
            this.face1.Image = faces[1 - 1];
            this.face2.Image = facesTwin[1 - 1];
            s = "#: " + labels[1 - 1] + " and #: " + labels[2 - 1] + "\tout: " + (predictions[1 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label1.Text = s;
            //Pair 2
            this.face3.Image = faces[2 - 1];
            this.face4.Image = facesTwin[2 - 1];
            s = "#: " + labels[3 - 1] + " and #: " + labels[4 - 1] + "\tout: " + (predictions[2 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label2.Text = s;
            //Pair 3
            this.face5.Image = faces[3 - 1];
            this.face6.Image = facesTwin[3 - 1];
            s = "#: " + labels[5 - 1] + " and #: " + labels[6 - 1] + "\tout: " + (predictions[3 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label3.Text = s;
            //Pair 4
            this.face7.Image = faces[4 - 1];
            this.face8.Image = facesTwin[4 - 1];
            s = "#: " + labels[7 - 1] + " and #: " + labels[8 - 1] + "\tout: " + (predictions[4 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label4.Text = s;
            //Pair 5
            this.face9.Image = faces[5 - 1];
            this.face10.Image = facesTwin[5 - 1];
            s = "#: " + labels[9 - 1] + " and #: " + labels[10 - 1] + "\tout: " + (predictions[5 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label5.Text = s;
            //Pair 6
            this.face11.Image = faces[6 - 1];
            this.face12.Image = facesTwin[6 - 1];
            s = "#: " + labels[11 - 1] + " and #: " + labels[12 - 1] + "\tout: " + (predictions[6 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label6.Text = s;
            //Pair 7
            this.face13.Image = faces[7 - 1];
            this.face14.Image = facesTwin[7 - 1];
            s = "#: " + labels[13 - 1] + " and #: " + labels[14 - 1] + "\tout: " + (predictions[7 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label7.Text = s;
            //Pair 8
            this.face15.Image = faces[8 - 1];
            this.face16.Image = facesTwin[8 - 1];
            s = "#: " + labels[15 - 1] + " and #: " + labels[16 - 1] + "\tout: " + (predictions[8 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label8.Text = s;
            //Pair 9
            this.face17.Image = faces[9 - 1];
            this.face18.Image = facesTwin[9 - 1];
            s = "#: " + labels[17 - 1] + " and #: " + labels[18 - 1] + "\tout: " + (predictions[9 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label9.Text = s;
            //Pair 10
            this.face19.Image = faces[10 - 1];
            this.face20.Image = facesTwin[10 - 1];
            s = "#: " + labels[19 - 1] + " and #: " + labels[20 - 1] + "\tout: " + (predictions[10 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label10.Text = s;
            //Pair 11
            this.face21.Image = faces[11 - 1];
            this.face22.Image = facesTwin[11 - 1];
            s = "#: " + labels[21 - 1] + " and #: " + labels[22 - 1] + "\tout: " + (predictions[11 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label11.Text = s;
            //Pair 12
            this.face23.Image = faces[12 - 1];
            this.face24.Image = facesTwin[12 - 1];
            s = "#: " + labels[23 - 1] + " and #: " + labels[24 - 1] + "\tout: " + (predictions[12 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label12.Text = s;
            //Pair 13
            this.face25.Image = faces[13 - 1];
            this.face26.Image = facesTwin[13 - 1];
            s = "#: " + labels[25 - 1] + " and #: " + labels[26 - 1] + "\tout: " + (predictions[13 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label13.Text = s;
            //Pair 14
            this.face27.Image = faces[14 - 1];
            this.face28.Image = facesTwin[14 - 1];
            s = "#: " + labels[27 - 1] + " and #: " + labels[28 - 1] + "\tout: " + (predictions[14 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label3.Text = s;
            //Pair 15
            this.face29.Image = faces[15 - 1];
            this.face30.Image = facesTwin[15 - 1];
            s = "#: " + labels[29 - 1] + " and #: " + labels[30 - 1] + "\tout: " + (predictions[15 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label15.Text = s;
            //Pair 16
            this.face31.Image = faces[16 - 1];
            this.face32.Image = facesTwin[16 - 1];
            s = "#: " + labels[31 - 1] + " and #: " + labels[32 - 1] + "\tout: " + (predictions[16 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label16.Text = s;
            //Pair 17
            this.face33.Image = faces[17 - 1];
            this.face34.Image = facesTwin[17 - 1];
            s = "#: " + labels[33 - 1] + " and #: " + labels[34 - 1] + "\tout: " + (predictions[17 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label17.Text = s;
            //Pair 18
            this.face35.Image = faces[18 - 1];
            this.face36.Image = facesTwin[18 - 1];
            s = "#: " + labels[35 - 1] + " and #: " + labels[36 - 1] + "\tout: " + (predictions[18 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label18.Text = s;
            //Pair 19
            this.face37.Image = faces[19 - 1];
            this.face38.Image = facesTwin[19 - 1];
            s = "#: " + labels[37 - 1] + " and #: " + labels[38 - 1] + "\tout: " + (predictions[19 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label19.Text = s;
            //Pair 20
            this.face39.Image = faces[20 - 1];
            this.face40.Image = facesTwin[20 - 1];
            s = "#: " + labels[39 - 1] + " and #: " + labels[40 - 1] + "\tout: " + (predictions[20 - 1] == 1 ? "MATCH" : "NO MATCH");
            this.label20.Text = s;

        }

    }
}
