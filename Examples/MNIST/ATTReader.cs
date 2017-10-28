using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace ATTFace
{
    public static class ATTReader
    {
        //public static List<ATTEntry> Load(string labelFile, string imageFile, int maxItem = -1)
        //{
        //    var label = LoadLabels(labelFile, maxItem);
        //    var images = LoadImages(imageFile, maxItem);

        //    if (label.Count == 0 || images.Count == 0)
        //    {
        //        return new List<ATTEntry>();
        //    }

        //    return label.Select((t, i) => new ATTEntry { Label = t, Image = images[i] }).ToList();
        //}

        public static List<ATTEntry> Load(string filename, int maxItem = -1)
        {

            /**
             * Images are in 40 different folders, with 10 images in each folder = 400 pictures.
             * Their paths are orl_faces\s1\1.pgm, 2.pgm, ... , 10.pgm
             *                 orl_faces\s2\1.pgm, 2.pgm, ... , 10.pgm
             *                 ...
             * This method should 
             * 
             */

            var result = new List<ATTEntry>();

            if (File.Exists(filename))
            {
                using (ZipArchive archive = ZipFile.OpenRead(filename))
                {
                    foreach (ZipArchiveEntry entry in archive.Entries)
                    {
                        //skip readme file.
                        if (entry.Name.Equals("README"))
                            continue;
                        //skip directories.
                        if (entry.Length == 0)
                            continue;

                        result.Add(MakeEntry(entry));
                    }
                }

            }
            
            return result;

            

            //var result = new List<byte[]>();

            //if (File.Exists(filename))
            //{



            //    using (var instream = new GZipStream(File.Open(filename, FileMode.Open), CompressionMode.Decompress))
            //    {





            //        using (var reader = new BinaryReader(instream))
            //        {
            //            var magicNumber = ReverseBytes(reader.ReadInt32());
            //            var numberOfImage = ReverseBytes(reader.ReadInt32());
            //            var rowCount = ReverseBytes(reader.ReadInt32());
            //            var columnCount = ReverseBytes(reader.ReadInt32());
            //            if (maxItem != -1)
            //            {
            //                numberOfImage = Math.Min(numberOfImage, maxItem);
            //            }

            //            for (var i = 0; i < numberOfImage; i++)
            //            {
            //                var image = reader.ReadBytes(rowCount * columnCount);
            //                result.Add(image);
            //            }
            //        }
            //    }
            //}

            //return result;
        }

        private static ATTEntry MakeEntry(ZipArchiveEntry item)
        {
            //Image 
            //Label
            //SetNum


            //Read in image byte data. pgm format
            //Code modified from James D. McCaffrey's blog
            // https://jamesmccaffrey.wordpress.com/2014/10/21/a-pgm-image-viewer-using-c/
            byte[] pixels;
            using (var reader = new BinaryReader(item.Open()))
            {
                string magic = NextNonCommentLine(reader);
                if (magic != "P5")
                    throw new Exception("Unknown magic number: " + magic);

                string widthHeight = NextNonCommentLine(reader);
                string[] tokens = widthHeight.Split(' ');
                int width = int.Parse(tokens[0]);
                int height = int.Parse(tokens[1]);

                string sMaxVal = NextNonCommentLine(reader);
                int maxVal = int.Parse(sMaxVal);

                pixels = reader.ReadBytes(width * height);
            }

            //Get label and setnum from entry's full name.
            string filename = item.FullName;
            string[] nametokens = filename.Split('/');
            int label = int.Parse(nametokens[nametokens.Length - 2].Remove(0, 1));
            //nametokens[nametokens.Length - 1].Trim('.', 'p', 'g', 'm');
            //int setnum = int.Parse(nametokens[nametokens.Length - 1].Remove(1));
            int setnum = int.Parse(nametokens[nametokens.Length - 1].Trim('.', 'p', 'g', 'm'));

            return new ATTEntry()
            {
                Image = pixels,
                Label = label,
                SetNum = setnum
            };
        }

        static string NextAnyLine(BinaryReader br)
        {
            string s = "";
            byte b = 0; // dummy
            while (b != 10) // newline
            {
                b = br.ReadByte();
                char c = (char)b;
                s += c;
            }
            return s.Trim();
        }

        static string NextNonCommentLine(BinaryReader br)
        {
            string s = NextAnyLine(br);
            while (s.StartsWith("#") || s == "")
                s = NextAnyLine(br);
            return s;
        }




        //private static List<int> LoadLabels(string filename, int maxItem = -1)
        //{
        //    var result = new List<int>();

        //    if (File.Exists(filename))
        //    {
        //        using (var instream = new GZipStream(File.Open(filename, FileMode.Open), CompressionMode.Decompress))
        //        {
        //            using (var reader = new BinaryReader(instream))
        //            {
        //                var magicNumber = ReverseBytes(reader.ReadInt32());
        //                var numberOfItem = ReverseBytes(reader.ReadInt32());
        //                if (maxItem != -1)
        //                {
        //                    numberOfItem = Math.Min(numberOfItem, maxItem);
        //                }

        //                for (var i = 0; i < numberOfItem; i++)
        //                {
        //                    result.Add(reader.ReadByte());
        //                }
        //            }
        //        }
        //    }

        //    return result;
        //}

        //private static int ReverseBytes(int v)
        //{
        //    var intAsBytes = BitConverter.GetBytes(v);
        //    Array.Reverse(intAsBytes);
        //    return BitConverter.ToInt32(intAsBytes, 0);
        //}


    }
}