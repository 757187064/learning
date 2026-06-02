import Foundation
import Vision
import ImageIO

if CommandLine.arguments.count < 2 {
    exit(2)
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    exit(3)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["zh-Hans", "en-US"]
request.minimumTextHeight = 0.012

let handler = VNImageRequestHandler(cgImage: image, options: [:])
try handler.perform([request])

let observations = (request.results ?? []).compactMap { obs -> (CGRect, String)? in
    guard let text = obs.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines),
          !text.isEmpty else {
        return nil
    }
    return (obs.boundingBox, text)
}

let ordered = observations.sorted {
    let dy = abs($0.0.midY - $1.0.midY)
    if dy > 0.025 { return $0.0.midY > $1.0.midY }
    return $0.0.minX < $1.0.minX
}

for (_, text) in ordered {
    print(text)
}
