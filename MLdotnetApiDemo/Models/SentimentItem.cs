﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLdotnetApiDemo.Models
{
    public class SentimentItem
    {
        public int Id { get; set; }
        public string Text { get; set; }
        public double CognitiveSentimentScore { get; set; }
        public bool MlNetSentimentScore { get; set; }
        public bool MlCustomSentimentScore { get; set; }
    }
}