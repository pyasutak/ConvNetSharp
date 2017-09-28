using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ATTFace
{
    public partial class DisplayImage : Form
    {
        public DisplayImage()
        {
            InitializeComponent();
        }



        public void DisplayData(byte[] img1, int lbl1, byte[] img2, int lbl2, String result)
        {
            //Bitmap bmp, bmp2;
            //using (var ms = new MemoryStream(img1))
            //using (var ms2 = new MemoryStream(img2))
            //{
            //    bmp = new Bitmap(ms);
            //    bmp2 = new Bitmap(ms2);
            //}


            this.pictureBox1.Image = ToBitmap(img1, 28, 28);
            this.pictureBox2.Image = ToBitmap(img2, 28, 28);
            this.label1.Text = "#: " + lbl1;
            this.label2.Text = "#: " + lbl2;
            this.resultLabel.Text = result;
        }

        public void DisplayData(Bitmap img1, int lbl1, Bitmap img2, int lbl2, String result)
        {
            //Bitmap bmp, bmp2;
            //using (var ms = new MemoryStream(img1))
            //using (var ms2 = new MemoryStream(img2))
            //{
            //    bmp = new Bitmap(ms);
            //    bmp2 = new Bitmap(ms2);
            //}


            this.pictureBox1.Image = img1;
            this.pictureBox2.Image = img2;
            this.label1.Text = "#: " + lbl1;
            this.label2.Text = "#: " + lbl2;
            this.resultLabel.Text = result;
        }

        public static Bitmap ToBitmap(byte[] pixels, int width, int height)
        {
            Bitmap bmp = new Bitmap(width, height);

            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    bmp.SetPixel(i, j, Color.FromArgb(pixels[i + j * width], pixels[i + j * width], pixels[i + j * width]));
                }
            }
            return bmp;
        }




    }
}
