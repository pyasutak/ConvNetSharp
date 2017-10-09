namespace ATTFace
{
    public class ATTEntry
    {
        public byte[] Image { get; set; }

        public int Label { get; set; }

        public int SetNum { get; set; }

        public override string ToString()
        {
            return "Label: " + this.Label + "_" + SetNum;
        }
    }
}