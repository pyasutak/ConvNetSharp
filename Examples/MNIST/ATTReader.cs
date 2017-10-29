using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace ATTFace
{
    public static class ATTReader
    {
        public static List<ATTEntry> Load(string filename, int maxItem = -1)
        {
            /**
             * Images are in 40 different folders, with 10 images in each folder = 400 pictures.
             * Their paths are orl_faces\s1\1.pgm, 2.pgm, ... , 10.pgm
             *                 orl_faces\s2\1.pgm, 2.pgm, ... , 10.pgm
             *                 ...
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
        }

        private static ATTEntry MakeEntry(ZipArchiveEntry item)
        {
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
        
    }
}