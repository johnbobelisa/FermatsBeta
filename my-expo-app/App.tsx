import './global.css'
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Alert,
  Dimensions,
  StatusBar,
  SafeAreaView,
  Platform,
  KeyboardAvoidingView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Line } from 'react-native-svg';

interface Hold {
  id: string;
  type: 'start' | 'regular' | 'finish';
  position: { x: number; y: number };
}

interface WallScale {
  pixelDistance: string;
  realDistanceCm: string;
}

interface ImageSize {
  width: number;
  height: number;
}

interface Position {
  x: number;
  y: number;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<ImageSize>({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState<ImageSize>({ width: 0, height: 0 });
  const [holds, setHolds] = useState<Hold[]>([]);
  const [selectedMarkerType, setSelectedMarkerType] = useState<'start' | 'regular' | 'finish'>('start');
  const [problemName, setProblemName] = useState<string>('');
  const [wallScale, setWallScale] = useState<WallScale>({ pixelDistance: '', realDistanceCm: '' });
  const [climberHeight, setClimberHeight] = useState<string>('');
  const [climberArmSpan, setClimberArmSpan] = useState<string>('');
  const [isScaleMode, setIsScaleMode] = useState<boolean>(false);
  const [scalePoints, setScalePoints] = useState<Position[]>([]);
  
  const holdIdCounter = useRef<number>(0);

  const handleImageUpload = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Sorry, we need camera roll permissions to make this work!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'images',
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      
      // Calculate display size - use more screen width
      const maxWidth = screenWidth - 32; // 16px padding on each side
      const maxHeight = screenHeight * 0.6; // Use up to 60% of screen height
      const ratio = Math.min(maxWidth / asset.width, maxHeight / asset.height);
      
      setImageSize({ width: asset.width, height: asset.height });
      setDisplaySize({ 
        width: asset.width * ratio, 
        height: asset.height * ratio 
      });
      setImage(asset.uri);
      setHolds([]);
      setScalePoints([]);
      holdIdCounter.current = 0;
    }
  };

  const getMarkerColor = (type: 'start' | 'regular' | 'finish'): string => {
    switch (type) {
      case 'start': return '#10b981';
      case 'finish': return '#f59e0b';
      case 'regular': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  const validateHolds = (): boolean => {
    const startHolds = holds.filter(h => h.type === 'start');
    const finishHolds = holds.filter(h => h.type === 'finish');
    const regularHolds = holds.filter(h => h.type === 'regular');

    if (startHolds.length < 1 || startHolds.length > 2) {
      Alert.alert('Validation Error', 'You need 1-2 start markers');
      return false;
    }
    if (finishHolds.length !== 1) {
      Alert.alert('Validation Error', 'You need exactly 1 finish marker');      
      return false;
    }
    if (regularHolds.length < 1) {
      Alert.alert('Validation Error', 'You need at least 1 regular marker');
      return false;
    }
    return true;
  };

  const handleImagePress = (event: any) => {
    if (!image) return;

    const { locationX, locationY } = event.nativeEvent;
    const x = (locationX * imageSize.width) / displaySize.width;
    const y = (locationY * imageSize.height) / displaySize.height;
    
    if (isScaleMode) {
      const newPoint: Position = { x, y };
      if (scalePoints.length === 0) {
        setScalePoints([newPoint]);
      } else if (scalePoints.length === 1) {
        const point1 = scalePoints[0];
        const distance = Math.sqrt(Math.pow(x - point1.x, 2) + Math.pow(y - point1.y, 2));
        setWallScale({ ...wallScale, pixelDistance: distance.toFixed(0) });
        setScalePoints([point1, newPoint]);
      } else {
        setScalePoints([newPoint]);
        setWallScale({ ...wallScale, pixelDistance: '' });
      }
      return;
    }

    // Check hold limits before adding
    const currentHolds = holds.filter(h => h.type === selectedMarkerType);
    
    if (selectedMarkerType === 'start' && currentHolds.length >= 2) {
      Alert.alert('Hold Limit', 'Maximum 2 start holds allowed');
      return;
    }
    
    if (selectedMarkerType === 'finish' && currentHolds.length >= 1) {
      Alert.alert('Hold Limit', 'Maximum 1 finish hold allowed');
      return;
    }

    const newHold: Hold = {
      id: `h${holdIdCounter.current++}`,
      type: selectedMarkerType,
      position: { x: Math.round(x), y: Math.round(y) }
    };

    setHolds([...holds, newHold]);
  };

  const removeHold = (holdId: string) => {
    setHolds(holds.filter(h => h.id !== holdId));
  };

  const generateBeta = async () => {
    if (!validateHolds()) return;
    
    if (!problemName || !wallScale.pixelDistance || !wallScale.realDistanceCm || !climberHeight || !climberArmSpan) {
      Alert.alert('Missing Information', 'Please fill in all fields');
      return;
    }

    const payload = {
      problem_name: problemName,
      wall: {
        scale: {
          pixelDistance: parseInt(wallScale.pixelDistance),
          realDistanceCm: parseInt(wallScale.realDistanceCm)
        }
      },
      climber: {
        heightCm: parseInt(climberHeight),
        armSpanCm: parseInt(climberArmSpan)
      },
      holds: holds
    };

    console.log('Sending to backend:', JSON.stringify(payload, null, 2));
    Alert.alert('Success', 'Beta generation request sent! Check console for payload.');
  };

  const MarkerTypeButton = ({ type }: { type: 'start' | 'regular' | 'finish' }) => (
    <TouchableOpacity
      onPress={() => setSelectedMarkerType(type)}
      className={`flex-1 py-3 px-4 rounded-xl mx-1 ${
        selectedMarkerType === type ? 'opacity-100' : 'opacity-70'
      }`}
      style={{ backgroundColor: getMarkerColor(type) }}
    >
      <Text className="text-white text-center font-medium capitalize">
        {type}
      </Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView className="flex-1 bg-slate-50">
      <StatusBar barStyle="dark-content" backgroundColor="#f8fafc" />
      
      {/* Navigation Bar */}
      <View className="bg-white/80 border-b border-blue-100 px-6 py-4">
        <View className="flex-row items-center">
          <LinearGradient
            colors={['#2563eb', '#3b82f6']}
            className="w-8 h-8 rounded-lg items-center justify-center"
          >
            <Text className="text-white font-bold text-sm">F</Text>
          </LinearGradient>
          <Text className="ml-3 text-xl font-bold text-blue-600">
            FermataBeta
          </Text>
        </View>
      </View>

      <KeyboardAvoidingView 
        className="flex-1"
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        <ScrollView 
          className="flex-1" 
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={{ paddingBottom: 40 }}
        >
          {/* Hero Section */}
          <LinearGradient
            colors={['#1e3a8a', '#1e40af', '#2563eb']}
            className="px-6 py-12"
          >
            <View className="max-w-3xl">
              <Text className="text-3xl font-bold text-white mb-8">
                Find Your Optimal Path
              </Text>
              <View className="space-y-6">
                <View className="flex-row items-start">
                  <View className="w-2 h-2 bg-blue-300 rounded-full mt-3 mr-4" />
                  <Text className="text-white text-base leading-relaxed flex-1">
                    <Text className="text-blue-200 font-semibold">Fermat's Principle:</Text> Light chooses the most efficient path through space
                  </Text>
                </View>
                <View className="flex-row items-start">
                  <View className="w-2 h-2 bg-blue-300 rounded-full mt-3 mr-4" />
                  <Text className="text-white text-base leading-relaxed flex-1">
                    <Text className="text-blue-200 font-semibold">Bouldering Beta:</Text> The optimal sequence of moves and body positions
                  </Text>
                </View>
                <View className="flex-row items-start">
                  <View className="w-2 h-2 bg-blue-300 rounded-full mt-3 mr-4" />
                  <Text className="text-white text-base leading-relaxed flex-1">
                    Our app finds the most efficient climbing path, just like light finds its optimal route!
                  </Text>
                </View>
              </View>
            </View>
          </LinearGradient>

          {/* Main Content */}
          <View className="p-6">
            <View className="items-center mb-12">
              <Text className="text-2xl font-bold text-gray-800 mb-4">Route Marker</Text>
              <Text className="text-gray-600 text-center leading-relaxed px-4">
                Upload your climbing wall image and mark the holds to generate optimal beta
              </Text>
            </View>
            
            {!image ? (
              <TouchableOpacity
                onPress={handleImageUpload}
                className="bg-white border-2 border-blue-200 rounded-2xl p-12 items-center mb-8 mx-2"
                style={{ borderStyle: 'dashed' }}
              >
                <View className="w-16 h-16 bg-blue-100 rounded-full items-center justify-center mb-6">
                  <Text className="text-2xl">📷</Text>
                </View>
                <Text className="text-gray-700 text-lg font-medium mb-2">
                  Upload climbing wall image
                </Text>
                <Text className="text-gray-500 text-sm">
                  JPG, PNG, or GIF • Max 10MB
                </Text>
              </TouchableOpacity>
            ) : (
              <View className="mb-8">
                <View className="relative bg-white rounded-2xl overflow-hidden shadow-lg border border-blue-100 self-center mb-8">
                  <TouchableOpacity onPress={handleImagePress} activeOpacity={1}>
                    <Image
                      source={{ uri: image }}
                      style={{ width: displaySize.width, height: displaySize.height }}
                      resizeMode="contain"
                    />
                  </TouchableOpacity>
                  
                  {/* Markers */}
                  {holds.map((hold: Hold) => (
                    <TouchableOpacity
                      key={hold.id}
                      onPress={() => removeHold(hold.id)}
                      className="absolute w-5 h-5 rounded-full border-2 border-white"
                      style={{
                        backgroundColor: getMarkerColor(hold.type),
                        left: (hold.position.x * displaySize.width) / imageSize.width - 10,
                        top: (hold.position.y * displaySize.height) / imageSize.height - 10,
                      }}
                    />
                  ))}

                  {/* Scale Points */}
                  {scalePoints.map((point, index) => (
                    <View
                      key={`scale-point-${index}`}
                      className="absolute w-6 h-6 bg-amber-500 rounded-full border-2 border-white items-center justify-center"
                      style={{
                        left: (point.x * displaySize.width) / imageSize.width - 12,
                        top: (point.y * displaySize.height) / imageSize.height - 12,
                      }}
                    >
                      <Text className="text-white text-xs font-bold">P{index + 1}</Text>
                    </View>
                  ))}
                  
                  {/* Scale line */}
                  {scalePoints.length === 2 && (
                    <Svg 
                      style={{ position: 'absolute', top: 0, left: 0 }}
                      width={displaySize.width} 
                      height={displaySize.height}
                    >
                      <Line
                        x1={(scalePoints[0].x * displaySize.width) / imageSize.width}
                        y1={(scalePoints[0].y * displaySize.height) / imageSize.height}
                        x2={(scalePoints[1].x * displaySize.width) / imageSize.width}
                        y2={(scalePoints[1].y * displaySize.height) / imageSize.height}
                        stroke="#f59e0b"
                        strokeWidth="3"
                        strokeDasharray="5,5"
                      />
                    </Svg>
                  )}
                </View>
                
                <TouchableOpacity
                  onPress={handleImageUpload}
                  className="bg-gray-600 py-4 px-8 rounded-xl self-center mb-8"
                >
                  <Text className="text-white font-medium">Change Image</Text>
                </TouchableOpacity>
              </View>
            )}

            {image && (
              <View className="space-y-8">
                {/* Scale Setting */}
                <View className="bg-white p-6 rounded-2xl shadow-sm border border-blue-100">
                  <View className="flex-row items-center mb-6">
                    <View className="w-2 h-2 bg-blue-500 rounded-full mr-3" />
                    <Text className="text-lg font-semibold text-gray-800">Scale Setting</Text>
                  </View>
                  
                  <TouchableOpacity
                    onPress={() => {
                      setIsScaleMode(!isScaleMode);
                      if (isScaleMode) {
                        // Clear scale points when exiting scale mode
                        setScalePoints([]);
                      }
                    }}
                    className={`py-4 px-6 rounded-xl mb-6 ${
                      isScaleMode ? 'bg-amber-500' : 'bg-blue-500'
                    }`}
                  >
                    <Text className="text-white text-center font-medium">
                      {isScaleMode ? '✓ Exit Scale Mode' : '📏 Set Scale (Click 2 points)'}
                    </Text>
                  </TouchableOpacity>
                  
                  <View className="flex-row space-x-4">
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-medium">Pixel Distance</Text>
                      <TextInput
                        value={wallScale.pixelDistance}
                        onChangeText={(text) => setWallScale({...wallScale, pixelDistance: text})}
                        placeholder="410"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-white text-base"
                      />
                    </View>
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-medium">Real Distance (cm)</Text>
                      <TextInput
                        value={wallScale.realDistanceCm}
                        onChangeText={(text) => setWallScale({...wallScale, realDistanceCm: text})}
                        placeholder="100"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-white text-base"
                      />
                    </View>
                  </View>
                </View>

                {/* Marker Controls */}
                {!isScaleMode && (
                  <View className="bg-white p-6 rounded-2xl shadow-sm border border-blue-100">
                    <View className="flex-row items-center mb-6">
                      <View className="w-2 h-2 bg-blue-500 rounded-full mr-3" />
                      <Text className="text-lg font-semibold text-gray-800">Marker Type</Text>
                    </View>
                    
                    <View className="flex-row mb-8 mx-1">
                      <MarkerTypeButton type="start" />
                      <MarkerTypeButton type="regular" />
                      <MarkerTypeButton type="finish" />
                    </View>
                    
                    <View className="bg-gray-50 rounded-xl p-6">
                      <View className="flex-row justify-around">
                        <View className="items-center">
                          <Text className="font-semibold text-emerald-600 text-lg">
                            {holds.filter(h => h.type === 'start').length}/2
                          </Text>
                          <Text className="text-gray-600 text-sm mt-1">Start</Text>
                        </View>
                        <View className="items-center">
                          <Text className="font-semibold text-blue-600 text-lg">
                            {holds.filter(h => h.type === 'regular').length}
                          </Text>
                          <Text className="text-gray-600 text-sm mt-1">Regular</Text>
                        </View>
                        <View className="items-center">
                          <Text className="font-semibold text-amber-600 text-lg">
                            {holds.filter(h => h.type === 'finish').length}/1
                          </Text>
                          <Text className="text-gray-600 text-sm mt-1">Finish</Text>
                        </View>
                      </View>
                    </View>
                  </View>
                )}

                {/* Problem Details */}
                <View className="bg-white p-6 rounded-2xl shadow-sm border border-blue-100">
                  <View className="flex-row items-center mb-6">
                    <View className="w-2 h-2 bg-blue-500 rounded-full mr-3" />
                    <Text className="text-lg font-semibold text-gray-800">Problem Details</Text>
                  </View>
                  
                  <View className="mb-6">
                    <Text className="text-gray-700 mb-3 font-medium">Problem Name</Text>
                    <TextInput
                      value={problemName}
                      onChangeText={setProblemName}
                      placeholder="RedV5"
                      className="border border-gray-200 rounded-xl px-4 py-4 bg-white text-base"
                    />
                  </View>

                  <View className="flex-row space-x-4">
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-medium">Climber Height (cm)</Text>
                      <TextInput
                        value={climberHeight}
                        onChangeText={setClimberHeight}
                        placeholder="170"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-white text-base"
                      />
                    </View>
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-medium">Arm Span (cm)</Text>
                      <TextInput
                        value={climberArmSpan}
                        onChangeText={setClimberArmSpan}
                        placeholder="170"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-white text-base"
                      />
                    </View>
                  </View>
                </View>

                {/* Generate Beta Button */}
                <View className="mt-8 mb-12">
                  <TouchableOpacity onPress={generateBeta}>
                    <LinearGradient
                      colors={['#2563eb', '#1d4ed8']}
                      className="py-5 px-8 rounded-2xl shadow-lg"
                    >
                      <View className="flex-row items-center justify-center">
                        <Text className="text-xl mr-3">🚀</Text>
                        <Text className="text-white text-lg font-semibold">Generate Beta</Text>
                      </View>
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}