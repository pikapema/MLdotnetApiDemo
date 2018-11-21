using Microsoft.AspNetCore.Mvc;

namespace MLdotnetApiDemo.Controllers
{
    [Route("api/MLNETmodel")]
    public class MLNETmodelController : Controller
    {
        static MLNETmodelController()
        {
        }

        [HttpPost]
        public int Post(string text)
        {
            System.Diagnostics.Trace.WriteLine("POST called");
            return MLNetTextSentiment.PredictSentiment(text);
            //return 1;
        }

    }
}
