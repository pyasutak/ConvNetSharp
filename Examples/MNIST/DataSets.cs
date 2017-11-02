using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Collections.Generic;

namespace ATTFace
{
    internal class DataSets
    {
        private const string urlATT = @"http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip";
        private const string attFolder = @"..\att\";
        private const string attDataSet = "att_faces.zip";
        
        public DataSet Train { get; set; }

        public DataSet Validation { get; set; }

        //public DataSet Test { get; set; }

        private void DownloadFile(string urlFile, string destFilepath)
        {
            if (!File.Exists(destFilepath))
            {
                try
                {
                    using (var client = new WebClient())
                    {
                        client.DownloadFile(urlFile, destFilepath);
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Failed downloading " + urlFile);
                    Console.WriteLine(e.Message);
                }
            }
        }
        
        public bool Load(int validationSize = 1000) //validationSize is not used.
        {
            Directory.CreateDirectory(attFolder);
            
            var attDataSetFilePath = Path.Combine(attFolder, attDataSet);
            
            // Download data set if needed
            Console.WriteLine("Downloading AT&T Face Data Set files...");
            DownloadFile(urlATT, attDataSetFilePath);
            
            // Load data
            Console.WriteLine("Loading the datasets...");
            //var train_images = ATTReader.Load(trainingLabelFilePath, trainingImageFilePath);
            //var testing_images = ATTReader.Load(testingLabelFilePath, testingImageFilePath);
            var load_images = ATTReader.Load(attDataSetFilePath);
            
            //split each face into validation.
            //Dataset has 40 faces, 10 versions of each.
            //Take three versions of each face for validation set.
            List<ATTEntry> train_images = new List<ATTEntry>();
            List<ATTEntry> valiation_images = new List<ATTEntry>();

            for (int i = 0; i < load_images.Count; i++)
            {
                if (i % 10 < 3)
                    valiation_images.Add(load_images[i]);
                else
                    train_images.Add(load_images[i]);
            }
            
            //train_images = load_images.GetRange(0, load_images.Count - validationSize);
            //valiation_images = load_images.GetRange(load_images.Count - validationSize, validationSize);

            //if (train_images.Count == 0 || valiation_images.Count == 0 || testing_images.Count == 0)
            if (train_images.Count == 0 || valiation_images.Count == 0)
            {
                Console.WriteLine("Missing ATT Data Set files.");
                Console.ReadKey();
                return false;
            }

            this.Train = new DataSet(train_images);
            this.Validation = new DataSet(valiation_images);
            //this.Test = new DataSet(testing_images);

            return true;
        }
    }
}