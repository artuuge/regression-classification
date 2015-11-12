{-# LANGUAGE TemplateHaskell #-}

import System.Random
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.HMatrix
import Control.Lens hiding ((<.>), (??))
import Data.Default
import Data.List.Split

----------

data Config = Config { _epsilonInit :: R
                     , _learningRate :: R
                     , _numIterations :: Int
                     }
  deriving (Show)
makeLenses ''Config

instance Default Config where
  def = Config { _epsilonInit = 0.001
               , _learningRate = 0.03
               , _numIterations = 300
               }

-- numIterations ~ 10/ learningRate

----------

vcons :: R -> Vector R -> Vector R
vcons x v = vjoin [vector [x], v]

forward :: Matrix R -> Vector R -> Vector R
forward theta x = app theta (vcons 1.0 x)

gradient1p :: Matrix R -> (Vector R, Vector R) -> Matrix R
gradient1p theta (x, y) =
  let a = forward theta x
   in mul (asColumn (a - y)) (asRow (vcons 1.0 x))

sumGradients :: Matrix R -> [(Vector R, Vector R)] -> Matrix R
sumGradients theta es =
  let initDelta = konst 0.0 (size theta) :: Matrix R
   in foldr (+) initDelta $ fmap (gradient1p theta) es

updateWeights :: R -> [(Vector R, Vector R)] -> Matrix R -> Matrix R
updateWeights alpha es theta =
  let ne = fromInteger . fromIntegral $ length es
      bigDelta = sumGradients theta es
   in theta + (bigDelta * (scalar ((-alpha)/ne)))

trainWeights :: Config -> [(Vector R, Vector R)] -> Matrix R -> Matrix R
trainWeights c es theta =
  (!! n) $ iterate (updateWeights alpha es) theta
    where
      n = c ^. numIterations
      alpha = c ^. learningRate

----------

newRandomVectors :: (R, R) -> Int -> Int -> IO [Vector R]
newRandomVectors r n d = do
  g <- newStdGen
  return $ fmap vector $ take n $ chunksOf d $ randomRs r g

newRandomExamples :: (R, R)
                  -> (R, R)
                  -> Matrix R
                  -> Int  
                  -> IO [(Vector R, Vector R)]
newRandomExamples r1 r2 mat numEx = do
  let (m, n) = size mat
  features <- newRandomVectors r1 numEx (n - 1)
  flucts <- newRandomVectors r2 numEx m
  let labels = zipWith f features flucts
  return $ zip features labels
    where 
      f = \x dy -> dy + app mat (vcons 1 x)

newRandomMatrix :: (R, R) -> (Int, Int) -> IO (Matrix R)
newRandomMatrix valRange (numRows, numCols) = do 
  g <- newStdGen
  return $
    matrix numCols $
    take (numRows * numCols) $
    randomRs valRange g

newRandomWeights :: Config -> (Int, Int) -> IO (Matrix R)
newRandomWeights c = newRandomMatrix (-eps, eps)
  where eps = c ^. epsilonInit

----------

underline :: String -> String
underline xs = xs ++ "\n" ++ (replicate (length xs) '-')

main :: IO ()
main = do

  let ndd = 8    -- number of displayed digits
  let nIn = 10
  let nOut = 4
  let myMat = matrix (nIn + 1) $ take (nOut * (nIn + 1)) [1..]
  let numExamples = 10000 

  putStrLn $ underline "Linear layer (multivariate linear regression):"
  putStrLn ""

  putStrLn $ "True weight matrix (" ++ (show nIn) ++ " inputs, " ++ (show nOut) ++  " outputs, the 1st column is the intercept):"
  disp 16 myMat
  putStrLn ""

  putStrLn $ "Random initialization of the weight matrix:"
  theta <- newRandomWeights def (size myMat) 
  disp ndd theta
  putStrLn ""

  putStrLn $ "Trained weight matrix (" ++ (show numExamples) ++ " random examples, " ++ (show $ def ^. numIterations) ++ " iterations):"
  es <- newRandomExamples (-10.0, 10.0) (-0.5, 0.5) myMat numExamples
  let theta' = trainWeights def es theta 
  disp ndd theta'
  putStrLn ""


  return ()

----------

